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
import time
import itertools

from eval_clip import eval_imagenet_zero_shot_cls
from utils import feature_vis
from utils import GradualWarmupScheduler
from caption_VLM import generate_caption

from dataset.datasets import load_training_data, loading_im3d, loading_imgnet
from model.model_ViTriLoss import VITriLoss
from model.build_model import load_model, load_model_multigpu
from config import get_opts
import open_clip

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

def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[cuda_device].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    # x = torch.cuda.FloatTensor(256, 1024, block_mem)
    x = torch.FloatTensor(256, 1024, block_mem).cuda(cuda_device)
    del x


opt = get_opts()
device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
for cuda_device in opt.gpu_ids:
    occumpy_mem(cuda_device)

model_text, model_image, model, preprocess = load_model_multigpu(device)
if opt.use_method == 'OPENAI':
  # tokenizer=clip.tokenize
  tokenizer=open_clip.tokenize
else:
  tokenizer=open_clip.tokenize
dataset, train_dataloader, IMGNET_train_dataloader = load_training_data(preprocess=preprocess, dataset_info_file=opt.training_info_path)
# dataset, train_dataloader = loading_im3d(preprocess=preprocess, dataset_info_file=opt.training_info_path[2])
# IMGNET_train_dataloader = loading_imgnet(preprocess=preprocess, dataset_info_file=opt.training_info_path[1])


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

optimizer = optim.AdamW(model.parameters(), lr=opt.lr,betas=(0.9,0.98),eps=1e-6, weight_decay=opt.weight_decay) #Psarams used from paper, the lr is smaller, more safe for fine tuning to new dataset
# modify T-max
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch,
                                                              eta_min=1e-7, last_epoch=-1)
# lr_scheduler = GradualWarmupScheduler(optimizer, start_lr=0, total_epoch=5, after_scheduler=scheduler_cosine)
                                                     

# for hard maximal
mode = torch.float16



def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

# def compute_knn_weight(feature, neighbors=5): # feature (each_obj_num, 512) —> weight[i] (each_obj_num, i)
#     weights = torch.zeros((feature.shape[0], feature.shape[1]), dtype=torch.float32).to(device)
#     for i in range(feature.shape[0]):
#         dist = torch.cdist(feature[i,:,:], feature[i,:,:])
#         _, indices = dist.topk(neighbors, largest=False, dim=-1)
#         weight = 1 / (torch.gather(dist, 1, indices).sum(dim=1) + 1e-8)
#         weights[i,:] = weight
#     return weights

# def compute_center_feature(clip_feature: torch.Tensor, method='knn') -> torch.Tensor:
#     clip_feature = clip_feature.to(torch.float32)
#     if method == 'knn':
#       mask = (clip_feature.sum(dim=2) != 0).unsqueeze(-1)
#       # valid_view_count = mask.sum(dim=1, keepdim=True).squeeze(-1)
#       weights = compute_knn_weight(clip_feature).unsqueeze(-1)
#       # 除去空缺值
#       weights = weights*mask
#       sum_weight = weights.sum(dim=1, keepdim=True).squeeze(-1)

#       weighted_feature_sum = (clip_feature * weights).sum(dim=1)
#       center_feature = weighted_feature_sum / (sum_weight + 1e-8)  # 为了避免除以0，加一个小常数
#     else:
#       mask = (clip_feature.sum(dim=2) != 0).float().unsqueeze(-1)
#       valid_view_count = mask.sum(dim=1, keepdim=True).squeeze(-1)
#       weighted_feature_sum = (clip_feature * mask).sum(dim=1)
#       center_feature = weighted_feature_sum / (valid_view_count + 1e-8)  # 为了避免除以0，加一个小常数

#     center_feature = F.normalize(center_feature, p=2, dim=1)
#     center_feature = center_feature.to(torch.float32)
#     return center_feature


def calculate_knn_weighted_feature_center(clip_feature: torch.Tensor, eps=1e-8):
    object_num, view_num, feature_dim = clip_feature.shape
    center_feature = torch.zeros((object_num, feature_dim)).type_as(clip_feature)

    for object_idx in range(object_num):
        object_features = clip_feature[object_idx]

        # Remove views with all zero features
        valid_views = object_features[~(object_features==0).all(1)]
        if len(valid_views) == 0:
            continue  # Skip if no valid views

        # Compute cosine similarity matrix
        cos_sim = torch.mm(valid_views, valid_views.t())

        # Calculate KNN weights
        weights = torch.zeros(len(valid_views)).type_as(clip_feature)
        for i in range(len(valid_views)):
            # Get indices of 5 nearest neighbors, excluding the view itself
            knn_indices = cos_sim[i].topk(6, largest=True).indices[1:]
            knn_avg = cos_sim[i][knn_indices].mean()
            weights[i] = 1 / (knn_avg + eps) if knn_avg != 0 else 0

        # Normalize weights
        weights /= weights.sum() if weights.sum() != 0 else 1

        # Calculate weighted feature center
        weighted_feature_center = (weights.unsqueeze(1) * valid_views).sum(dim=0)
        center_feature[object_idx] = weighted_feature_center

    return center_feature

def compute_sim_batch(total_features, all_features, all_object_ids, center_features, batch_size=256):
  similarities = torch.zeros(total_features, dtype=mode).to(device)

  for i in range(0, total_features, batch_size):
    end_idx = min(i + batch_size, total_features)
    features_batch = all_features[i:end_idx]
    object_ids_batch = all_object_ids[i:end_idx]
    center_features_batch = center_features[object_ids_batch]

    sim_batch = F.cosine_similarity(features_batch, center_features_batch, dim=1)
    similarities[i:end_idx] = sim_batch

  return similarities


def hard_maximal(train_dataloader, dataset, model, method=opt.center_method, K=opt.K):

  with torch.no_grad():
    feature_dim = 768 if opt.clip_model_name == 'ViT-L/14' or opt.clip_model_name == 'ViT-L/14@336px' or  opt.clip_model_name == 'ViT-L-14' or opt.clip_model_name == 'ViT-L-14-quickgelu' else 512

    # print(feature_dim)
    label_features = torch.zeros((dataset.object_num, 100, feature_dim), dtype=mode).to(device) # for feature center compute
    label_counters = torch.zeros(dataset.object_num, dtype=torch.long).to(device)

    total_features = len(dataset)
    all_features = torch.zeros((total_features, feature_dim), dtype=mode).to(device)
    all_object_ids = torch.zeros(total_features, dtype=torch.long).to(device)
    all_image_ids = torch.zeros(total_features, dtype=torch.long).to(device)

    feature_idx = 0
    model.eval()
    print("compute_clip_feature")
    for batch_idx, batch in enumerate(train_dataloader):
      images,texts,object_id,img_id = batch
      images = images.to(device)
      object_id = object_id.to(device)
      img_id = img_id.to(device)

      logits_per_image = model(images).to(mode)
      logits_per_image = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
      
      num_images = len(object_id)
      all_features[feature_idx:feature_idx+num_images] = logits_per_image
      all_object_ids[feature_idx:feature_idx+num_images] = object_id
      all_image_ids[feature_idx:feature_idx+num_images] = img_id
      feature_idx += num_images
      
      for i, obj_id in enumerate(object_id):
        label_index = label_counters[obj_id.item()]
        label_features[obj_id, label_index, :] = logits_per_image[i, :]
        label_counters[obj_id] += 1

      if batch_idx % opt.log_interval == 0:
              print(' compute center feature:[{}/{} ({:.0f}%)]'.format(
                  (batch_idx+1) * len(images), len(train_dataloader.dataset),
                        100. * (batch_idx+1) / len(train_dataloader)))

    print("compute_center_feature")
    # center_features = compute_center_feature(label_features, method=method)
    center_features = calculate_knn_weighted_feature_center(label_features)
    print("compute K hard viewpoint")
    # similarities = F.cosine_similarity(all_features, center_features[all_object_ids])
    sim = compute_sim_batch(total_features, all_features, all_object_ids, center_features, batch_size=256)
    # select hard example
    hard_examples = defaultdict(list)
    for idx, (similarity, img_id) in enumerate(zip(sim, all_image_ids)):
        obj_id = all_object_ids[idx].item()
        hard_examples[obj_id].append((similarity.item(), img_id.item()))

    for obj_id, examples in hard_examples.items():
        examples.sort(key=lambda x: x[0])  # 根据相似度排序
        hard_examples[obj_id] = [im_id for _, im_id in examples[:K]]

    del all_features
    del all_object_ids
    del all_image_ids
    del label_features
    del label_counters
    del sim
    torch.cuda.empty_cache()
  return center_features, hard_examples

def combine_and_shuffle_data(images, images_clean, texts, texts_clean):
    combined_images = torch.cat([images, images_clean], dim=0)
    combined_texts = torch.cat([texts, texts_clean], dim=0)
    indices = torch.randperm(combined_images.size(0))

    shuffled_images = combined_images[indices]
    shuffled_texts = combined_texts[indices]

    return shuffled_images, shuffled_texts

def infinite_loader(dataloader):
    while True:
        for data in dataloader:
            yield data
        # 数据集结束，重新开始
        # dataloader.dataset.shuffle()

if __name__ == '__main__':
  # add your own code to track the training progress.
  # before training
  # debug:
  print(opt)
  top1s = []
  top5s = []

  # for pair in opt.test_data_label_path:
  #   dataset_path, label_path, gt_label_path = pair
  #   top1, top5 = eval_imagenet_zero_shot_cls(model, device, preprocess, dataset_path=dataset_path, label_path=label_path, gt_label_path=gt_label_path, tokenizer=tokenizer)
  #   top1s.append(top1)
  #   top5s.append(top5)
  #   torch.cuda.empty_cache()

  iter_num = 0
  iter_dataloader_IMGNET = infinite_loader(IMGNET_train_dataloader)
  for epoch in tqdm(range(opt.epoch)):
    # clip_feature = compute_clip_feature(train_dataloader, dataset, model)
    # # feature_vis(clip_feature, epoch)

    # center_feature = compute_center_feature(clip_feature, method=opt.center_method)
    # hard_examples = hard_maximal_hash(center_feature, train_dataloader, model, K=opt.K)
    #--------------------------------
    center_feature, hard_examples = hard_maximal(train_dataloader, dataset, model_image, method=opt.center_method, K=opt.K)
    center_feature = center_feature.to(torch.float32)

    # print('\n center_features: \n', center_feature)
    # center_feature_list = center_feature.tolist()
    # with open(f'center_feature_{epoch}.json', 'w') as f:
    #   json.dump(center_feature_list, f)

    # 创建干净数据的迭代器
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        # batch_IMGNET = next(iter_dataloader_IMGNET)

        images,texts,object_id,img_id = batch 
        # images_clean, texts_clean = batch_IMGNET
      
        images = images.to(device)
        texts = texts.to(device)
        object_id = object_id.to(device)
        img_id = img_id.to(device)

        # images_clean = images_clean.to(device)
        # texts_clean = texts_clean.to(device) # [32, 1, 77]
        # print(images.shape)
        # modify print
        # for name, param in model.named_parameters():
        #   if torch.isnan(param).any():
        #     print(f"\n Parameter with NaN values: {name} \n")

        # logits_per_image, logits_per_text = model(images, texts)
        # shuffled_images, shuffled_texts = combine_and_shuffle_data(images, images_clean, texts, texts_clean)
        image_embedding = model_image(images)
        text_embedding = model_text(texts)
        logit_scale = model.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        constrat_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        selected_indices = [idx for idx, im_id in enumerate(img_id) if im_id.item() in hard_examples[object_id[idx].item()]]
        if not selected_indices:  # 如果当前批次没有所需的图像，跳过此批次
          vi_loss = torch.tensor(0.0)
        else:
          selected_images = images[selected_indices]
          selected_obj_id = object_id[selected_indices]
          image_features = model_image(selected_images)
          image_features = image_features / image_features.norm(dim=-1, keepdim=True)

          vi_loss = VITriLoss(image_features, selected_obj_id, center_feature)
          
          # similarities = F.cosine_similarity(image_features, center_feature[selected_obj_id])
          # vi_loss = 1.0 - similarities.mean()

        total_loss = constrat_loss + opt.lamba*vi_loss

        total_loss.backward()
        # modify print
        # if torch.isnan(constrat_loss).any():
        # for param in model.parameters():
        #   if param.requires_grad:
        #     print(param.grad)

        # print('\n center_features: \n', center_feature[selected_obj_id])
        # print('\n image_embedding: \n', image_embedding)
        # print('\n text_embedding: \n', text_embedding)
        # print('\n logits_per_image: \n', logits_per_image)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print(total_loss)
        # if device == "cpu":
        #   optimizer.step()
        # else : 
        #   convert_models_to_fp32(model)
        #   optimizer.step()
        #   clip.model.convert_weights(model)
        #   convert_models_to_fp16(model.transformer_module)
        optimizer.step()
        # for name, param in model.named_parameters():
        #   if torch.isnan(param).any():
        #     print(f"\n Parameter with NaN values: {name} \n")

        if batch_idx % opt.log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttoatal_Loss: {:.6f}\tconstrat_Loss:{:.6f}\tvi_Loss:{:.6f}'.format(
                  epoch, (batch_idx+1) * len(images), len(train_dataloader.dataset),
                        100. * (batch_idx+1) / len(train_dataloader), total_loss.item(), constrat_loss.item(), vi_loss.item()))
        # if iter_num % opt.save_interval == 0:
        #     torch.save(model, f"../src/model_checkpoint/re-exp/{opt.exp}_epoch={epoch}_iter={iter_num}.pt") #just change to your preferred folder/filename
        iter_num += 1
    lr_scheduler.step()
    # del iter_dataloader_IMGNET
    # del hard_examples
    # torch.cuda.empty_cache()

    # eval clean data(imagenet val)
    for pair in opt.test_data_label_path:
      dataset_path, label_path, gt_label_path = pair
      top1, top5 = eval_imagenet_zero_shot_cls(model, device, preprocess, dataset_path=dataset_path, label_path=label_path, gt_label_path=gt_label_path, tokenizer=tokenizer)
      top1s.append(top1)
      top5s.append(top5)
    
    torch.save(model, f"../src/model_checkpoint/re-exp/{opt.exp}_epoch={epoch}.pt") #just change to your preferred folder/filename
    torch.cuda.empty_cache()
  print(top1s)
  print(top5s)
  
  # model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training
  # checkpoint = torch.load("../src/model_checkpoint/model_epoch=20.pt")

  # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
  # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
  # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
  # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

  # model.load_state_dict(checkpoint['model_state_dict'])
