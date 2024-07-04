import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import clip
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
from dataset.datasets import load_eval_data
from model.build_model import load_pretrain, load_model_multigpu
from config import get_opts

import open_clip

opt = get_opts()

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(classnames, templates, model, device, tokenizer):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def eval_imagenet_zero_shot_cls(model, device, preprocess, dataset_path, label_path, gt_label_path, tokenizer=clip.tokenize):

    val_loader = load_eval_data(data_path=dataset_path, preprocess=preprocess) 

    print(f"zero-shot classification for '{dataset_path}'")
    imagenet_classes = []
    with open(label_path) as f:
        for line in f.readlines():
            imagenet_classes.append(line.split("\n")[0])

    gt_labels = []
    with open(gt_label_path) as f:
        for line in f.readlines():
            gt_labels.append(line.strip())
    
    # 创建映射：从测试集标签到imagenet_classes的索引
    label_to_idx = {label: idx for idx, label in enumerate(imagenet_classes)}
    gt_indices = [label_to_idx[label] for label in gt_labels]

    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model, device, tokenizer)
    results = []

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = torch.tensor([gt_indices[t] for t in target], device=device)
            # print("shaaaape:", images.shape)
            # print("detype:", images.dtype)
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

    return top1, top5


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_images_from_folder(folder, device):
    """加载指定文件夹内的所有图像并返回它们的列表"""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据你的文件类型调整
            path = os.path.join(folder, filename)
            image = Image.open(path).convert("RGB")
            images.append(image)
    return images

def get_clip_embeddings(images, model, preprocess):
    """使用CLIP模型获取图像列表的嵌入"""
    images_preprocessed = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(images_preprocessed).float()
    return embeddings


def main(device, model, preprocess, gt_label_path, dir_path, label_id, tokenizer=clip.tokenize):

    imagenet_classes = []
    with open(gt_label_path) as f:
        for line in f.readlines():
            imagenet_classes.append(line.split("\n")[0])
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model, device, tokenizer)

    topk=(1, 3)
    top1, top3, n = 0., 0., 0.

    for folder in tqdm(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            images = load_images_from_folder(folder_path, device)
            embeddings = get_clip_embeddings(images, model, preprocess)
            mean_embedding = embeddings.mean(dim=0).unsqueeze(0)
            mean_embedding /= mean_embedding.norm(dim=-1, keepdim=True)
            logits = 100. * mean_embedding @ zeroshot_weights

            pred = logits.topk(max(topk), 1, True, True)[1].t()
            correct = pred.eq(label_id.view(1, -1).expand_as(pred))

            acc1, acc3 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
            top1 += acc1
            top3 += acc3
            n += 1
    
    top1 = (top1 / n) * 100
    top3 = (top3 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-3 accuracy: {top3:.2f}")

    
if __name__ == '__main__':
     # evaulate zero-shot calssification use model checkpoint

    device = f"cuda:6" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    # openai clip
    # model, preprocess = load_pretrain(device, model_name='ViT-L/14@336px', checkpoint=None, is_ovt_model=False)
    # # model = torch.load('/data1/yinpeng.dong/Omniview-Tuning/src/model_checkpoint/re-exp/ovt_clip_vitb16_test_epoch=2.pt').to(device)
    # tokenizer = clip.tokenize
    
    # open clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    # ('ViT-B-32', pretrained='laion2b_s34b_b79k') ('ViT-B-16', pretrained='laion2b_s34b_b88k') ('ViT-L-14', pretrained='laion2b_s32b_b82k')
    # model = torch.load('/data1/yinpeng.dong/Omniview-Tuning/src/model_checkpoint/re-exp/ovt_openclip_vitb16_epoch=3_iter=35000_.pt', map_location=device)
    model = model.to(device)
    print("model's params for training:", count_parameters(model))

    tokenizer = open_clip.tokenize

    main(device, model, preprocess, 
        gt_label_path='/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imv+.txt', 
        dir_path='/data1/yinpeng.dong/Omniview-Tuning/rebuttal/shoe', 
        label_id=torch.tensor([70], device=device), tokenizer=tokenizer)

    # OVT-clip
    # model, preprocess = load_pretrain(device, model_name='ViT-B/16', checkpoint=None, is_ovt_model=False)
    # model = torch.load('/data1/yinpeng.dong/Omniview-Tuning/src/model_checkpoint/ovt_clip_vitb16_K10_epoch=4.pt').to(device)
    # tokenizer = clip.tokenize

    # eva_clip
    # model_name = "EVA02-CLIP-L-14"  # EVA-B-16: EVA02-CLIP-B-16; EVA-L-14: EVA02-CLIP-L-14; EVA-L-14-336: EVA02-CLIP-L-14-336
    # pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
    # model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    # tokenizer = get_tokenizer(model_name)
    # model = model.to(device)

    # METACLIP
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='metaclip_fullcc')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
    # # B-16: ViT-B-16-quickgelu; B-32: ViT-B-32-quickgelu; L-14: ViT-L-14-quickgelu
    # model = torch.load('/data1/yinpeng.dong/Omniview-Tuning/src/model_checkpoint/re-exp/ovt_metaclip_vitl14_epoch=1_iter=20000.pt').to(device)
    # model = model.to(device)
    # tokenizer = open_clip.tokenize


    # model, preprocess = load_pretrain(device, model_name="ViT-B/16", checkpoint='/data1/yinpeng.dong/Omniview-Tuning/src/model_checkpoint/ovt_clip_vitb16_epoch=9.pt', is_ovt_model=True)

