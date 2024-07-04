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

seed_value = 2012
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda:1" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
EPOCH = 30
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512
lamba = 0
log_interval = 10
K = 5
center_method = 'knn'
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training
# model = torch.nn.DataParallel(model)

transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt, object_id, transform):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.object_id = torch.LongTensor(object_id)
        self.object_num = int(torch.max(self.object_id))+1
        self.transform = transform

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
          image = self.transform(image)
        image = preprocess(image) # Image from PIL module
        title = self.title[idx]
        object_id = self.object_id[idx]
        img_id = idx
        return image, title, object_id, img_id

# use your own data
data_path = []
data_txt = []
object_id = []


im3d = './GMFool_dataset_mvclip'
imagenet_train = './imagenet/train'
imagenet_classes = []
with open('./labels.txt') as f:
  for line in f.readlines():
      imagenet_classes.append(line.split("\n")[0])


t = 0
_id = 0
for objects in tqdm(os.listdir(im3d)):
  for img in os.listdir(os.path.join(im3d, objects)):
    img_path = os.path.join(os.path.join(im3d, objects), img)
    data_path.append(img_path)
    data_txt.append(f"a photo of {objects}")
    object_id.append(_id)
    t += 1
    if t % 100 == 0:
      _id += 1

# captions = generate_caption(data_path, device)

with open('captions_im3d.json', 'r') as f:
    captions = json.load(f)
# print(len(captions))

# print(object_id)
# print(data_path)

# print(im3d_data_path)

# list_image_path = ['folder/image1.jpg','folder2/image2.jpg'] 
# list_txt = ['description for image1.jpg' , 'description for image2.jpg']
dataset = image_title_dataset(data_path, captions, object_id, transform=transform)
kwargs = {'num_workers': 10, 'pin_memory': True} if torch.cuda.is_available() else {}

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, **kwargs) #Define your own dataloader
# 准备加上随机旋转和平移仿射变换

imagenet_valdir = './imagenet/val'
imagenet_val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(imagenet_valdir, preprocess),batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

imagenetvplus_dir = './imagenet-v+'
imagenetvplus_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(imagenetvplus_dir, preprocess), batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

imagenetv_dir = './imagenet-v'
imagenetv_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(imagenetv_dir, preprocess), batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_ii = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH,
                                                              eta_min=0, last_epoch=-1)

# # before training
eval_imagenet_zero_shot_cls(model, imagenet_val_loader, device, datasetname='imagenet')
eval_imagenet_zero_shot_cls(model, imagenetvplus_loader, device, datasetname='imagenet-v+')
eval_imagenet_zero_shot_cls(model, imagenetv_loader, device, datasetname='imagenet-v')

def compute_clip_feature(train_dataloader, dataset, model):
  with torch.no_grad():
    # 在epoch开始时，用于计算所有样本的CLIP特征，返回特征tensor(object_id*N*128)}
    label_features = torch.zeros((dataset.object_num, 100, 512))
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
    weights = torch.zeros((feature.shape[0], feature.shape[1])).to(device)
    for i in range(feature.shape[0]):
        dist = torch.cdist(feature[i,:,:], feature[i,:,:])
        _, indices = dist.topk(neighbors, largest=False, dim=-1)
        weight = 1 / (torch.gather(dist, 1, indices).sum(dim=1) + 1e-8)
        weights[i,:] = weight
    return weights

def compute_center_feature(clip_feature: torch.Tensor, method='knn') -> torch.Tensor:
    if method == 'knn':
      mask = (clip_feature.sum(dim=2) != 0).float().unsqueeze(-1)
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


  
if __name__ == '__main__':
  # add your own code to track the training progress.
  for epoch in tqdm(range(EPOCH)):
    clip_feature = compute_clip_feature(train_dataloader, dataset, model)
    # print(clip_feature)
    feature_vis(clip_feature, epoch)

    center_feature = compute_center_feature(clip_feature, method=center_method)
    hard_examples = hard_maximal_hash(center_feature, train_dataloader, model, K=K)

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

          similarities = F.cosine_similarity(image_features, center_feature[selected_obj_id])
          vi_loss = 1.0 - similarities.mean()

        total_loss = constrat_loss + lamba*vi_loss

        total_loss.backward()
        # print(total_loss)
        if device == "cpu":
          optimizer.step()
        else : 
          convert_models_to_fp32(model)
          optimizer.step()
          clip.model.convert_weights(model)
        lr_scheduler.step()

        if batch_idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttoatal_Loss: {:.6f}\tconstrat_Loss:{:.6f}\tvi_Loss:{:.6f}'.format(
                  epoch, (batch_idx+1) * len(images), len(train_dataloader.dataset),
                        100. * (batch_idx+1) / len(train_dataloader), total_loss.item(), constrat_loss.item(), vi_loss.item()))

    # eval clean data(imagenet val)
    eval_imagenet_zero_shot_cls(model, imagenet_val_loader, device, datasetname='imagenet')
    # eval adv_viewpoint data(imagenet-v+/imagenet-v)
    eval_imagenet_zero_shot_cls(model, imagenetvplus_loader, device, datasetname='imagenet-v+')
    eval_imagenet_zero_shot_cls(model, imagenetv_loader, device, datasetname='imagenet-v')

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"./model_checkpoint/model_epoch={epoch}.pt") #just change to your preferred folder/filename


  model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training
  checkpoint = torch.load("model_checkpoint/model_epoch=20.pt")

  # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
  checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
  checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
  checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

  model.load_state_dict(checkpoint['model_state_dict'])