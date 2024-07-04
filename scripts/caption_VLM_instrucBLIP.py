import torch
from PIL import Image
import math
import json
from lavis.models import load_model_and_preprocess
import torch.nn as nn 
from tqdm import tqdm
from dataset.datasets import load_training_data
from model.build_model import load_model
from argparse import ArgumentParser
# device_ids = [0, 1, 2, 3]
from config import get_opts
import os
CUDA_LAUNCH_BLOCKING=1

opt = get_opts()

# 读取JSON文件
def read_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# 从字典列表中提取path和label列表
def extract_path_label(data):
    paths = [item['path'] for item in data]
    labels = [item['label'] for item in data]
    return paths, labels

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
    max_mem = int(total * 0.7)
    block_mem = max_mem - used
    # x = torch.cuda.FloatTensor(256, 1024, block_mem)
    x = torch.FloatTensor(256, 1024, block_mem).cuda(cuda_device)
    del x


def generate_caption(img_list, label_list, device, model, vis_processors):
    captions = []
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    batch_size = 64
    print("generate caption..")
    for batch_idx in tqdm(range(math.ceil(len(img_list)/batch_size))):
        if batch_idx != math.ceil(len(img_list)/batch_size)-1:
            img_list_part = img_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
            label_list_part = label_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            img_list_part = img_list[batch_idx*batch_size:]
            label_list_part = label_list[batch_idx*batch_size:]

        images = [vis_processors["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device) for img_path in img_list_part]
        images = torch.cat(images, dim=0)
        prompts = [f"write a short description for the image." for label in label_list_part]
        caption = model.generate({"image": images, "prompt": prompts}, length_penalty=-1.0, max_length=32)
        for cap in caption:
            captions.append(cap)
            print(cap)

    # for i in range(len(img_list)):
    #     raw_image = Image.open(img_list[i]).convert("RGB")
    #     images = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    #     prompts = f'Write a description for the image, noting that the object of the image is a {label_list[i]}.'
    #     print('imgs', img_list[i])
    #     print('prompts', prompts)
    #     caption = model.generate({"image": images, "prompt": prompts})
    #     print('results', caption)

    with open(f'/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/vqa_caption_use/caption_mvcap_wo_guide_{opt.f}.json', 'w') as f:
        json.dump(captions, f)

if __name__ == '__main__':
    device = f"cuda:{opt.fs}" if torch.cuda.is_available() else "cpu"
    # [{'path': ..., 'label':...},{},...]
    # occumpy_mem(opt.fs)

    data = read_json_file('/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/vqa_caption_use/mvcap_path_label.json')
    paths, labels = extract_path_label(data)
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
    # paths = paths[opt.b:opt.s]
    # labels = labels[opt.b:opt.s]
    # generate_caption(paths, labels, device, model, vis_processors)
    captions = generate_caption(paths[opt.b:opt.s], labels[opt.b:opt.s], device, model, vis_processors)
    
    # prepare the image
    # raw_image = Image.open(paths[0]).convert("RGB")
    # images = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # label = 'coffeepot'
    # prompts = f'Write a description for the image, noting that the object of the image is a {label}.'
    # print('imgs', paths[0])
    # print('prompts', prompts)
    # caption = model.generate({"image": images, "prompt": prompts})
    # print(caption)