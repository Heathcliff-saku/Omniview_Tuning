<div align="center">
  <h1>Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models</h1>
  <p>
    <a href="https://heathcliff-saku.github.io/">Shouwei Ruan</a>, 
    <a href="https://ml.cs.tsinghua.edu.cn/~yinpeng/">Yinpeng Dong</a>, 
    Hanqing Liu, Yao Huang, 
    <a href="https://www.suhangss.me/">Hang Su</a> and 
    <a href="https://sites.google.com/site/xingxingwei1988/">Xingxing Wei</a>.
  </p>
</div>
<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/63fc4751a3c067e62899a3a1/uRW0xd5mLDkc_YHh1073-.png" width="20%">
</div>
<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/pdf/2404.12139" style="margin: 0 10px;">
    <img src="https://img.shields.io/badge/Paper-Read-blue" alt="paper">
  </a>
  <a href="你的权重链接" style="margin: 0 10px;">
    <img src="https://img.shields.io/badge/Weight-Download-green?logo=huggingface" alt="weight">
  </a>
  <a href="https://huggingface.co/datasets/RSW233/MVCap-4M" style="margin: 0 10px;">
    <img src="https://img.shields.io/badge/Dataset-Download-yellow?logo=huggingface" alt="dataset">
  </a>
  <a href="https://github.com/Heathcliff-saku/Omniview_Tuning" style="margin: 0 10px;">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="code">
  </a>
</p>

This repo releases the **Code** of paper: **"Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models" (ECCV2024)**


Vision-Language Pre-training (VLP) models like CLIP have
achieved remarkable success in computer vision and particularly demonstrated superior robustness to distribution shifts of 2D images. However,
their robustness under **3D viewpoint variations** is still limited, which can
hinder the development for real-world applications. This paper successfully addresses this concern while keeping VLPs’ original performance by
breaking through two primary obstacles: 1) the scarcity of training data
and 2) the suboptimal fine-tuning paradigms. To combat data scarcity,
we build the Multi-View Caption (MVCap) dataset — a comprehensive
collection of over four million multi-view image-text pairs across more
than 100K objects, providing more potential for VLP models to develop
generalizable viewpoint-invariant representations. To address the limitations of existing paradigms in performance trade-offs and training efficiency, we design a novel fine-tuning framework named Omniview-Tuning
(OVT). Specifically, OVT introduces a Cross-Viewpoint Alignment objective through a minimax-like optimization strategy, which effectively
aligns representations of identical objects from diverse viewpoints without causing overfitting. Additionally, OVT fine-tunes VLP models in a
parameter-efficient manner, leading to minimal computational cost.

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/63fc4751a3c067e62899a3a1/QHuetkvOi2iEJUxKjWouU.png" width="70%">
</div>

## 0. Quick Start
- clone this repo:
```
git clone https://github.com/Heathcliff-saku/Omniview_Tuning.git
```
- install dependents: (we recommend using torch>=2.1.2 and cuda=12.x)
```
cd Omniview_tuning
pip install -r requirements.txt
```

## 1. Data Prepare
The dataset we provide consists of two parts: the MVCap-4M (training data) and the viwepoint-related downstream evaluation dataset, the source files can be downloaded via [our huggingface dataset repo](https://huggingface.co/datasets/RSW233/MVCap-4M) and extracted in the following format:

```
-- Omniview_tuning/
   -- dataset_source/
      -- labels/gt_labels/
      ... 
      -- metadata.json
      -- metadata_imgnet.json
      -- im3d/
      -- mvimgnet/
      -- views/
      ...
      -- imagenet-1k/
         -- train/
         -- val/
      -- imagenet-v/
      -- imagenet-v+/
```
then, in `scripts/config.py`, you should 
- replace the `--training_info_path` to [`path/to/your/metadata.json`, `path/to/your/metadata_imgnet.json`]
- replace the `--test_data_label_path` to [..., [`path/to/your/testdataset.json`, ..., ...],...]

**Note**: If you get a path-related error during runtime, e.g., file not found error. you may need to change the `path` in metadata.json to absolute path fromat.

### 1.1 Multi-View Caption Dataset (MVCap-4M)
MVCap is a large-scale dataset tailored for viewpoint invariance researches of Vison-Language Pretraining (VLP) models, comprising over 4.6 million multi-view image-text pairs across more than 100K objects. It contains the following parts:
- **metadata.json**：Stores the `path`, `caption`, `obj_id` and `img_id` sequence corresponding to each image sample of MVCap. The structures are looks like:
```
...
{
    "path": "./views/54cadb86f3db4aa6920f673aeff0d1e3/026.png",
    "caption": "The rocking chair in the image is made of metal and has a green cushion on it.",
    "obj_id": 3177,
    "img_id": 317726
},
...
```
- **source multi-view image**:
We sampled source multi viewpoint images from three existing 3D datasets：
  - Objavers-80k：Stores in subfolder `views.zip`
  - IM3D: Stores in subfolder `im3ds.zip`
  - MVImgNet: Stores in subfolder `mvimgnets.zip`

### 1.2 ImageNet-V & ImageNet-V+
The `IM-V` / `IM-V+` are both OOD datasets for benchmarking viewpoint robustness/invariance of visual recognition. the `IM-V` it's generated by [viewfool (NIPS2022)](https://arxiv.org/pdf/2210.03895), and has 10,000 renderings of 100 objects with images of size 400*400. The `IM-V+` is a larger OOD viewpoint benchmark, including 100K adversarial viewpoint samples captured by GMVFool on IM3D, which is proposed by [VIAT (ICCV2023)](https://arxiv.org/pdf/2307.10235).


## 2. Pretrain Weight

## 3. Evaluating

## 4. Omniview-Tuning



## :innocent: Citation

If you find our work useful, please consider citing our paper:
```
@article{Ruan2024Omniview,
  title={Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models},
  author={{Shouwei Ruan, Yinpeng Dong, Hanqing Liu, Yao Huang, Hang Su, Xingxing Wei}},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
and welcome to to refer to our previous work in Viewpoint Robustness/Invariance studies

```
@inproceedings{ruan2023towards,
  title={Towards viewpoint-invariant visual recognition via adversarial training},
  author={Ruan, Shouwei and Dong, Yinpeng and Su, Hang and Peng, Jianteng and Chen, Ning and Wei, Xingxing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4709--4719},
  year={2023}
}
```
```
@article{dong2022viewfool,
  title={Viewfool: Evaluating the robustness of visual recognition to adversarial viewpoints},
  author={Dong, Yinpeng and Ruan, Shouwei and Su, Hang and Kang, Caixin and Wei, Xingxing and Zhu, Jun},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36789--36803},
  year={2022}
}
```

## :satisfied: Contact Us!

- <showueiruan@buaa.edu.cn>
- <dongyinpeng@mail.tsinghua.edu.cn>
