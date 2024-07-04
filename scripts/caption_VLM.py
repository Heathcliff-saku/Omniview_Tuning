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

opt = get_opts()

def save_list_to_txt(data_list, file_name):
    """
    将列表保存到txt文件
    """
    with open(file_name, 'w') as file:
        for item in data_list:
            file.write("%s\n" % item)

def read_txt_to_list(file_name):
    """
    从txt文件读取内容到列表
    """
    with open(file_name, 'r') as file:
        return [line.strip() for line in file]


def generate_caption(img_list, device):
    captions = []
    model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device)
    batch_size = 32
    # model = nn.DataParallel(model, device_ids=device_ids)
    # model = model.to(device)
    print("generate caption..")
    for batch_idx in tqdm(range(math.ceil(len(img_list)/batch_size))):
        if batch_idx != math.ceil(len(img_list)/batch_size)-1:
            img_list_part = img_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            img_list_part = img_list[batch_idx*batch_size:]
        images = [vis_processors["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device) for img_path in img_list_part]
        images = torch.cat(images, dim=0)
        caption = model.generate({"image": images})
        for cap in caption:
            captions.append(cap)
            print(cap)
    with open(f'/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/caption_obj_{opt.f}.json', 'w') as f:
        json.dump(captions, f)
    return captions


if __name__ == '__main__':

    device = f"cuda:{opt.f}" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = load_model(device)
    # paths = load_training_data(data_path="/data1/yinpeng.dong/shouwei-mvclip/MVCLIP_project/rendering_objaverse_obj/objaverse-rendering/scripts/views", 
    #                             caption_path=None, 
    #                             preprocess=preprocess)
    # save_list_to_txt(paths, '/data1/yinpeng.dong/shouwei-mvclip/MVCLIP_project/mvclip/label_&_captions/path_of_obj.txt')
    paths = read_txt_to_list('/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/path_of_obj.txt')

    captions = generate_caption(paths[opt.b:opt.s], device)
