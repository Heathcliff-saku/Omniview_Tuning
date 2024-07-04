# Omniview-Tuning

## 1.building Multi-View Datasets

### 1.1 Objaverse 

- 从tag中筛选100类内符合的uids，首先用GPT4为每个类别扩充标签存在`download_objaverse_obj\labels_expend.txt`中，然后根据tag检索保存对应的uid，去重，人工筛选（每个.obj渲染出一个图像筛选）

- `./download_objaverse_obj/select_uid.py`

- **(√) 使用openshape数据集的标签检索**
- 筛选数据:
```
python .\download_objaverse_obj\openshape-demo-support\openshape\demo\retrieval.py
```
- 筛选后的物体uid:`download_objaverse_obj\matched_uids_final.txt` 数量: 24495
- 下载物体:
```
cd download_objaverse_obj
python download.py
```

### 1.2 MVImageNet
### 1.3 IM3D


## 2. Rendering for Objaverse

## 3. 重构代码

```
-- mvclip
    -- dataset 
        .. datasets.py
    -- model
        .. model_viformer.py
        .. model_VITriLoss.py

    -- model_ckeckpoint

    -- label_&_captions
        .. .json
        .. .txt
        ...,...
    
    .. caption_VLM.py
    .. config.py
    .. eval_clip.py
    .. train_viformer.py
    .. utils.py


```

## nohup 运行命令
```
    conda activate shouwei-mvclip

    nohup python -u train_viformer.py > running_log/debug.txt 2>&1 &

```


## 实验记录

```
对于组件的消融分析 
                                    IM            IM-V            IM-V+
0. viformer w/o viloss      |  69.04  88.58   57.45  74.62    59.07  82.70   
1. viformer w viloss        |  66.26  86.68   57.42  74.76    64.59  88.73
2. LoRA+viformer w/o viloss |  75.16  91.98   59.97  77.15    57.54  80.40
3. LoRA+viformer w viloss   |  




```
