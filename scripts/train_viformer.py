# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import clip
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import defaultdict
import random
import numpy as np

from tqdm import tqdm
import json

from eval_clip import eval_imagenet_zero_shot_cls
from utils import feature_vis
from caption_VLM import generate_caption

from dataset.datasets import load_training_data
from model.model_ViTriLoss import VITriLoss
from model.build_model import load_model
from config import get_opts

seed_value = 2012
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad != None:
            p.grad.data = p.grad.data.float() 

def convert_models_to_fp16(model): 
    for p in model.parameters(): 
        p.data = p.data.half()
        if p.grad != None:
            p.grad.data = p.grad.data.half()

opt = get_opts()
device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

model, preprocess = load_model(device)
dataset, train_dataloader = load_training_data(preprocess=preprocess, dataset_info_file=opt.training_info_path[0])

# if generate caption needed
# generate_caption(paths,device)

# if device == "cpu":
#   model.float()
# else :
#   clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float32

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_ii = nn.CrossEntropyLoss()
VITriLoss = VITriLoss(margin=opt.margin)

optimizer = optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9,0.98),eps=1e-6, weight_decay=opt.weight_decay) #Psarams used from paper, the lr is smaller, more safe for fine tuning to new dataset
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch,
                                                              eta_min=0, last_epoch=-1)

def compute_clip_feature(train_dataloader, dataset, model):
  with torch.no_grad():
    # 在epoch开始时，用于计算所有样本的CLIP特征，返回特征tensor(object_id*N*128)}
    label_features = torch.zeros((dataset.object_num, 100, 512), dtype=torch.float32)
    label_counters = torch.zeros(dataset.object_num, dtype=torch.long)

    print("compute_clip_feature")
    model.eval()
    for batch_idx, batch in enumerate(train_dataloader):
      images,texts,object_id,img_id = batch 
      images= images.to(device)
      object_id = object_id.to(device)

      logits_per_image = model.encode_image(images)
      for i in range(len(object_id)):
        object_id_ = object_id[i].item()
        logits_per_image_ = logits_per_image[i,:].squeeze(0)  # Squeezing to (128,) size
        # print(logits_per_image.shape)
        # print(logits_per_image_.shape)
        label_index = label_counters[object_id_]
        label_features[object_id_, label_index, :] = logits_per_image_
        label_counters[object_id_] += 1
      
      label_features = label_features.to(device)
    return label_features

def compute_knn_weight(feature, neighbors=5): # feature (each_obj_num, 512) —> weight[i] (each_obj_num, i)
    weights = torch.zeros((feature.shape[0], feature.shape[1]), dtype=torch.float32).to(device)
    for i in range(feature.shape[0]):
        dist = torch.cdist(feature[i,:,:], feature[i,:,:])
        _, indices = dist.topk(neighbors, largest=False, dim=-1)
        weight = 1 / (torch.gather(dist, 1, indices).sum(dim=1) + 1e-8)
        weights[i,:] = weight
    return weights

def compute_center_feature(clip_feature: torch.Tensor, method='knn') -> torch.Tensor:
    clip_feature = clip_feature.to(torch.float32)
    if method == 'knn':
      mask = (clip_feature.sum(dim=2) != 0).half().unsqueeze(-1)
      # valid_view_count = mask.sum(dim=1, keepdim=True).squeeze(-1)
      weights = compute_knn_weight(clip_feature).unsqueeze(-1)
      # 除去空缺值
      weights = weights*mask
      sum_weight = weights.sum(dim=1, keepdim=True).squeeze(-1)

      weighted_feature_sum = (clip_feature * weights).sum(dim=1)
      center_feature = weighted_feature_sum / (sum_weight + 1e-8)  # 为了避免除以0，加一个小常数
    else:
      mask = (clip_feature.sum(dim=2) != 0).float().unsqueeze(-1)
      valid_view_count = mask.sum(dim=1, keepdim=True).squeeze(-1)
      weighted_feature_sum = (clip_feature * mask).sum(dim=1)
      center_feature = weighted_feature_sum / (valid_view_count + 1e-8)  # 为了避免除以0，加一个小常数

    center_feature = F.normalize(center_feature, p=2, dim=1)
    center_feature = center_feature.to(torch.float32)
    return center_feature

def hard_maximal_hash(center_feature: torch.Tensor, train_dataloader: DataLoader, model, K: int) -> dict:
  print("compute K hard viewpoint")
  hard_examples = defaultdict(list)
  with torch.no_grad():
    model.eval()
    for batch_idx, batch in enumerate(train_dataloader):
      images,_,object_id,img_id = batch
      images = images.to(device)
      object_id = object_id.to(device)
      img_id = img_id.to(device)

      image_features = model.encode_image(images)
      similarities = F.cosine_similarity(image_features, center_feature[object_id])

      for obj_id, sim, im_id in zip(object_id, similarities, img_id):
        hard_examples[obj_id.item()].append((sim.item(), im_id.item()))
    
    for obj_id, examples in hard_examples.items():
      examples.sort(key=lambda x: x[0])  # 根据余弦相似性升序排列
      hard_examples[obj_id] = [im_id for _, im_id in examples[:K]]
    
  return hard_examples


def hard_maximal(train_dataloader, dataset, model, method=opt.center_method, K=opt.K):
  with torch.no_grad():
    print("compute_clip_feature")
    label_features = torch.zeros((dataset.object_num, 100, 512), dtype=torch.float32) # for feature center compute
    label_counters = torch.zeros(dataset.object_num, dtype=torch.long)
    
    total_features = sum(len(batch[0]) for batch in train_dataloader)
    all_features = torch.zeros((total_features, 512), dtype=torch.float32).to(device)
    all_object_ids = torch.zeros(total_features, dtype=torch.long).to(device)
    all_image_ids = torch.zeros(total_features, dtype=torch.long).to(device)
    feature_idx = 0
    model.eval()

    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
      images,texts,object_id,img_id = batch 
      images = images.to(device)
      object_id = object_id.to(device)
      img_id = img_id.to(device)

      logits_per_image = model.encode_image(images)
      for i in range(len(object_id)):
        object_id_ = object_id[i].item()
        logits_per_image_ = logits_per_image[i,:].squeeze(0)  # Squeezing to (128,) size

        label_index = label_counters[object_id_]
        label_features[object_id_, label_index, :] = logits_per_image_
        label_counters[object_id_] += 1

        all_features[feature_idx] = logits_per_image_
        all_object_ids[feature_idx] = object_id_
        all_image_ids[feature_idx] = img_id[i]
        feature_idx += 1

    label_features = label_features.to(device)
    print("compute_center_feature")
    center_features = compute_center_feature(label_features, method=method)
    print("compute K hard viewpoint")
    similarities = F.cosine_similarity(all_features, center_features[all_object_ids])
    # select hard example
    hard_examples = defaultdict(list)
    for idx, (similarity, img_id) in enumerate(zip(similarities, all_image_ids)):
        obj_id = all_object_ids[idx].item()
        hard_examples[obj_id].append((similarity.item(), img_id.item()))

    for obj_id, examples in hard_examples.items():
            examples.sort(key=lambda x: x[0])  # 根据相似度排序
            hard_examples[obj_id] = [im_id for _, im_id in examples[:K]]

  return center_features, hard_examples



  
if __name__ == '__main__':
  # add your own code to track the training progress.
  # before training
  top1s = []
  top5s = []

  for pair in opt.eval_data_label_path:
    dataset_path, label_path = pair
    top1, top5 = eval_imagenet_zero_shot_cls(model, device, preprocess, dataset_path=dataset_path, label_path=label_path)
    top1s.append(top1)
    top5s.append(top5)

  for epoch in tqdm(range(opt.epoch)):
    # clip_feature = compute_clip_feature(train_dataloader, dataset, model)
    # # feature_vis(clip_feature, epoch)

    # center_feature = compute_center_feature(clip_feature, method=opt.center_method)
    # hard_examples = hard_maximal_hash(center_feature, train_dataloader, model, K=opt.K)

    center_feature, hard_examples = hard_maximal(train_dataloader, dataset, model, method=opt.center_method, K=opt.K)

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        images,texts,object_id,img_id = batch 
      
        images= images.to(device)
        texts = texts.to(device)
        object_id = object_id.to(device)
        img_id = img_id.to(device)
        
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        constrat_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        selected_indices = [idx for idx, im_id in enumerate(img_id) if im_id.item() in hard_examples[object_id[idx].item()]]

        if not selected_indices:  # 如果当前批次没有所需的图像，跳过此批次
          vi_loss = torch.tensor(0.0)
        else:
          selected_images = images[selected_indices]
          selected_obj_id = object_id[selected_indices]
          image_features = model.encode_image(selected_images)
          vi_loss = VITriLoss(image_features, selected_obj_id, center_feature)
          # similarities = F.cosine_similarity(image_features, center_feature[selected_obj_id])
          # vi_loss = 1.0 - similarities.mean()

        total_loss = constrat_loss + opt.lamba*vi_loss

        total_loss.backward()
        # print(total_loss)
        # if device == "cpu":
        #   optimizer.step()
        # else : 
        #   convert_models_to_fp32(model)
        #   optimizer.step()
        #   clip.model.convert_weights(model)
        #   convert_models_to_fp16(model.transformer_module)
        optimizer.step()
        lr_scheduler.step()

        if batch_idx % opt.log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttoatal_Loss: {:.6f}\tconstrat_Loss:{:.6f}\tvi_Loss:{:.6f}'.format(
                  epoch, (batch_idx+1) * len(images), len(train_dataloader.dataset),
                        100. * (batch_idx+1) / len(train_dataloader), total_loss.item(), constrat_loss.item(), vi_loss.item()))

    # eval clean data(imagenet val)
    for pair in opt.eval_data_label_path:
      dataset_path, label_path = pair
      top1, top5 = eval_imagenet_zero_shot_cls(model, device, preprocess, dataset_path=dataset_path, label_path=label_path)
      top1s.append(top1)
      top5s.append(top5)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"../src/model_checkpoint/model_epoch={epoch}.pt") #just change to your preferred folder/filename

  print(top1s)
  print(top5s)
  
  model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training
  checkpoint = torch.load("../src/model_checkpoint/model_epoch=20.pt")

  # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
  checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
  checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
  checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

  model.load_state_dict(checkpoint['model_state_dict'])