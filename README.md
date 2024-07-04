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

This repo releases the **MVCap-4M dataset** introduced in our paper: **"Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models" (ECCV2024)**

Multi-View Caption (MVCap-4M) is a large-scale dataset tailored for viewpoint invariance researches of Vison-Language Pretraining (VLP) models, comprising over 4.6 million multi-view image-text pairs across more than 100K objects. To assemble a diverse collection of multi-view image-text pairs, we amalgamate various 3D assets with real-world multi-view data. This process involves an extensive selection and rendering of multi-view images from existing datasets. We then utilize a Vision Large Language Model (VLLM) for automated caption generation to obtain semantically rich textual descriptions without extensive manual efforts. To ensure category consistency across varying viewpoints in the generated captions, we implement a category-guided prompting strategy, which maintains accuracy in textual descriptions for different viewpoints of the same object or scene.

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/63fc4751a3c067e62899a3a1/QHuetkvOi2iEJUxKjWouU.png" width="70%">
</div>

## Data Release

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
- **source multi-view image**
We sampled source multi viewpoint images from three existing 3D datasets：
  - Objavers-80k：Stores in subfolder `/views`
  - IM3D: Stores in subfolder `/im3d`
  - MVImgNet: Stores in subfolder `/mvimgnet`

## Citation

If you find our work useful, please consider citing our paper:
```
@article{Ruan2024Omniview,
  title={Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models},
  author={{Shouwei Ruan, Yinpeng Dong, Hanqing Liu, Yao Huang, Hang Su, Xingxing Wei}},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Contact Us!

- <showueiruan@buaa.edu.cn>
- <dongyinpeng@mail.tsinghua.edu.cn>
