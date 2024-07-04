"""
dataset and dataloader for mvclip training
load_training_data()
load_eval_data()

training data should prepare in above format:

-- dataset_name (im3d/objaverse/..)
    -- object_00
        -- 001.png
        ...,...
        -- 099.png
    -- object_01
    ...,...
    -- object_nn

-- caption_file (im3d.json/...)
"""

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
import open_clip


from config import get_opts
opt = get_opts()

seed_value = 2012
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if opt.use_method == 'OPENAI':
  tokenizer=clip.tokenize
else:
  tokenizer=open_clip.tokenize

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt, object_id, transform, preprocess):

        self.image_path = list_image_path
        self.title  = tokenizer(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.object_id = torch.LongTensor(object_id)
        self.object_num = int(torch.max(self.object_id))+1
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
          image = self.transform(image)
        image = self.preprocess(image) # Image from PIL module
        # print(image.shape)
        title = self.title[idx]
        object_id = self.object_id[idx]
        img_id = idx
        return image, title, object_id, img_id


class image_title_dataset_use_info_file(Dataset):
    def __init__(self, dataset_info, transform, preprocess):

        self.image_path = [item['path'] for item in dataset_info if 'path' in item]
        self.title  = tokenizer([item['caption'] for item in dataset_info if 'caption' in item]) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.object_id = torch.LongTensor([item['obj_id'] for item in dataset_info if 'obj_id' in item])
        self.object_num = int(torch.max(self.object_id))+1
        self.transform = transform
        self.preprocess = preprocess

        self.transform
        print('done loading MVCAP data')

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
          image = self.transform(image)
        image = self.preprocess(image) # Image from PIL module
        title = self.title[idx]
        object_id = self.object_id[idx]
        img_id = idx
        return image, title, object_id, img_id


class image_title_dataset_use_info_file_imgnet(Dataset):
    def __init__(self, dataset_info, transform, preprocess):
        self.image_path = [item['path'] for item in dataset_info if 'path' in item]
        self.title  = tokenizer([item['caption'] for item in dataset_info if 'caption' in item]) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.transform = transform
        self.preprocess = preprocess
        print('done loading IMGNET data')

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
          image = self.transform(image)
        image = self.preprocess(image) # Image from PIL module
        title = self.title[idx]
        return image, title


def load_training_data(data_path=None, caption_path=None, preprocess=None, dataset_info_file=None):
    
    if dataset_info_file != None:
        with open(dataset_info_file[0], 'r') as file:
            dataset_info = json.load(file)
        MVCAP_dataset = image_title_dataset_use_info_file(dataset_info, transform=transform, preprocess=preprocess)
        kwargs = {'num_workers': opt.num_worker, 'pin_memory': True} if torch.cuda.is_available() else {}
        MVCAP_train_dataloader = DataLoader(MVCAP_dataset, batch_size=opt.batch_size,shuffle=True, persistent_workers=True, **kwargs)

        with open(dataset_info_file[1], 'r') as file:
            dataset_info = json.load(file)
        dataset = image_title_dataset_use_info_file_imgnet(dataset_info, transform=transform, preprocess=preprocess)
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        IMGNET_train_dataloader = DataLoader(dataset, batch_size=opt.batch_size_IMGNET,shuffle=True, persistent_workers=True, **kwargs)

        return MVCAP_dataset, MVCAP_train_dataloader, IMGNET_train_dataloader

    else:
        paths = []
        object_id = []
        t = 0
        _id = 0
        for objects in tqdm(sorted(os.listdir(data_path))):
            for img in sorted(os.listdir(os.path.join(data_path, objects))):
                img_path = os.path.join(os.path.join(data_path, objects), img)
                paths.append(img_path)
                # print(img_path)
                object_id.append(_id)
                t += 1
                if t % 100 == 0:
                    _id += 1

        
        if caption_path != None:
            with open(caption_path, 'r') as f:
                captions = json.load(f)

            dataset = image_title_dataset(paths, captions, object_id, transform=transform, preprocess=preprocess)
            kwargs = {'num_workers': opt.num_worker, 'pin_memory': True} if torch.cuda.is_available() else {}
            train_dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=True, persistent_workers=True, **kwargs)

            return dataset, train_dataloader
            
        else:
            return paths

def load_eval_data(data_path=None, preprocess=None):

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    val_dataloader = torch.utils.data.DataLoader(
      datasets.ImageFolder(data_path, preprocess),batch_size=opt.test_batch_size, shuffle=True, **kwargs)
    
    return val_dataloader


def loading_im3d(preprocess=None, dataset_info_file=None):
    with open(dataset_info_file, 'r') as file:
        dataset_info = json.load(file)
    dataset = image_title_dataset_use_info_file(dataset_info, transform=transform, preprocess=preprocess)
    kwargs = {'num_workers': opt.num_worker, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=True, persistent_workers=True, **kwargs)

    return dataset, dataloader

def loading_imgnet(preprocess=None, dataset_info_file=None):
    with open(dataset_info_file, 'r') as file:
        dataset_info = json.load(file)
    dataset = image_title_dataset_use_info_file_imgnet(dataset_info, transform=transform, preprocess=preprocess)
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    IMGNET_train_dataloader = DataLoader(dataset, batch_size=opt.batch_size_IMGNET,shuffle=True, persistent_workers=True, **kwargs)

    return IMGNET_train_dataloader