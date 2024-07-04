from argparse import ArgumentParser

def parse_list(option_string):
    return [int(item.strip()) for item in option_string.split(',')]

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--exp', type=str, default='ovt_clip_vitb16')
    # training
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--gpu_ids', type=parse_list, default=[4,5,6,7]) # if multigpu
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_IMGNET', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--num_worker', type=int, default=36)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--lamba', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # CLIP
    parser.add_argument('--use_method', type=str, default="OPENAI") # META #OPENCLIP
    parser.add_argument('--clip_model_name', type=str, default="ViT-B/16")
    # VIformer
    parser.add_argument('--use_viformer', type=bool, default=False)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    # LoRA
    parser.add_argument('--use_Lora', type=bool, default=False)

    # full param training
    parser.add_argument('--full_param_training', type=bool, default=False)

    # hard_maximal & ViTriLoss
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--center_method', type=str, default='knn')
    parser.add_argument('--margin', type=float, default=0.3)

    # path
    parser.add_argument('--training_info_path', type=list, default=['/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/vqa_caption_use/MVCAP_DATASET_Description_Clean_wo_guide.json',
                                                                    '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/vqa_caption_use/IMGNET_1K_DATASET_Description.json',
                                                                    '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/vqa_caption_use/IM3D_DATASET_Description.json'])
    parser.add_argument('--training_data_path', type=list, default=['/data1/yinpeng.dong/shouwei-dataset/im3d'])
    parser.add_argument('--training_caption_path', type=list, default=['/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/captions_im3d.json'])

    parser.add_argument('--eval_data_label_path', type=list, default=[['/data1/yinpeng.dong/shouwei-dataset/imagenet/val','/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imnet.txt'],
                                                                ['/data1/yinpeng.dong/shouwei-dataset/imagenet-v', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels_imagenet_v.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imv.txt'],
                                                                ['/data1/yinpeng.dong/shouwei-dataset/imagenet-v+', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imv+.txt'],
                                                                ['/data1/yinpeng.dong/shouwei-dataset/objectnet', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_objnet.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_objnet.txt'],
                                                                ['/data1/yinpeng.dong/shouwei-dataset/ood-cv', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_oodcv.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_oodcv.txt']])
    # parser.add_argument('--eval_data_label_path', type=list, default=[['C:/Users/86181/Desktop/rubbish/GMFooldataset_blender/111111111/ImageNet-V+', 'C:/Users/86181/Desktop/MVCLIP-cvpr2024/MVCLIP_project/mvclip/label_&_captions/labels.txt']])
    # must match with '--eval_data_path' one by one
    # 
    parser.add_argument('--test_data_label_path', type=list, 
    default=[['/data1/yinpeng.dong/shouwei-dataset/imagenet/val','/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imnet.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/ImageNet-1k','/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/cifar-100-test/test', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label_cifar.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label_cifar.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/ImageNet-v2','/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-ske/sketch', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-o', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_o_gt_label.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-r', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_r_gt_label.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ood-cv-full', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/gt_label_oodcv.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/imagenet-v', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels_imagenet_v.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imv.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/imagenet-v+', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_imv+.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/ood-cv', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/labels.txt', '/data1/yinpeng.dong/Omniview-Tuning/src/label_&_captions/gt_label/gt_label_oodcv.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/03_multi_view_testset/MIRO', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/03_multi_view_testset/miro_gt_label.txt'],
        ])
    
    parser.add_argument('--test_data_label_path_2', type=list, 
    default=[ 
            ['/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/cifar-100-test/test', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label_cifar.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label_cifar.txt'],
            ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-ske/sketch', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/00_clean_testset/gt_label.txt'],
             ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-a', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_a_gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_a_gt_label.txt'],
             ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-r', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_r_gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_r_gt_label.txt']
            ])

    # parser.add_argument('--test_data_label_path_2', type=list, 
    # default=[ 
    #         ['/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/ImageNet-a', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_a_gt_label.txt', '/data1/yinpeng.dong/shouwei-dataset/04_2d_ood_testset/im_a_gt_label.txt'],
    #         ])


    # caption_gen
    parser.add_argument('--b', type=int, default=0)
    parser.add_argument('--s', type=int, default=100)
    parser.add_argument('--f', type=int, default=0)
    parser.add_argument('--fs', type=int, default=0)
    
    return parser.parse_args()
