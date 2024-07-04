from .model_viformer import VIformer, CLIPWithVIformer, MLPFeatureTransform
from .model_NaiveMultiHeadAttention import PlainMultiHeadAttention
from config import get_opts
import clip
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
import open_clip

opt = get_opts()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad != None:
            p.grad.data = p.grad.data.float() 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_training_layers(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)


def load_model(device):
    clip_model, preprocess = clip.load(opt.clip_model_name,device=device,jit=False)
    convert_models_to_fp32(clip_model)
    # clip_model = clip_model.float()
    # use naive multihead atten to replace nn.multiheadatten in visual
    for module in clip_model.visual.transformer.resblocks:
        new_module = PlainMultiHeadAttention(embed_dim=module.attn.embed_dim, num_heads=module.attn.num_heads)
        new_module.set_parameters(module.attn)
        module.attn = new_module
    # print(clip_model)

    model_text = TextCLIP(clip_model)
    model_image = ImageCLIP(clip_model)

    # print(model_image)
    # print([key for key, _ in model_image.named_parameters()])

    if opt.use_Lora:
        lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, target_modules=['qkv'])
        model_image = get_peft_model(model_image, lora_config)

    feature_dim = clip_model.visual.output_dim
    viformer = VIformer(feature_dim=feature_dim, num_transformer_layers=opt.num_transformer_layers, nhead=opt.nhead, dim_feedforward=2048).to(device)

    # MLPFeatureTransform = MLPFeatureTransform(feature_dim).to(device)
    # Create the combined model
    model = CLIPWithVIformer(clip_model, viformer)
    print(model)
    
    return model, preprocess


def load_model_multigpu(device):
    
    if opt.use_method == "OPENAI":
        # clip_model, preprocess = clip.load(opt.clip_model_name,device=device,jit=False)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(opt.clip_model_name, pretrained='openai')
        clip_model = clip_model.to(device)
    elif opt.use_method == "OPENCLIP":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(opt.clip_model_name, pretrained='laion2b_s34b_b88k') # laion2b_s34b_b79k
        clip_model = clip_model.to(device)
    else:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(opt.clip_model_name, pretrained='metaclip_fullcc')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
        clip_model = clip_model.to(device)

    convert_models_to_fp32(clip_model)
    # use naive multihead atten to replace nn.multiheadatten in visual
    for module in clip_model.visual.transformer.resblocks:
        new_module = PlainMultiHeadAttention(embed_dim=module.attn.embed_dim, num_heads=module.attn.num_heads)
        new_module.set_parameters(module.attn)
        module.attn = new_module
    # print(clip_model)

    model_text = TextCLIP(clip_model)
    model_image = ImageCLIP(clip_model)

    # print(model_image)
    # print([key for key, _ in model_image.named_parameters()])
    if opt.full_param_training:
        for param in clip_model.parameters():
            param.requires_grad = True
    else:
        if opt.use_Lora:
            lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, target_modules=['qkv'])
            model_image = get_peft_model(model_image, lora_config)
        if opt.use_viformer:
            feature_dim = clip_model.visual.output_dim
            viformer = VIformer(feature_dim=feature_dim, num_transformer_layers=opt.num_transformer_layers, nhead=opt.nhead, dim_feedforward=2048).to(device)
            clip_model = CLIPWithVIformer(clip_model, viformer).to(device)

    model_text = TextCLIP(clip_model)
    model_image = ImageCLIP(clip_model)

    print("model", clip_model)
    print(f'training mode: full_params:{opt.full_param_training}+LoRA:{opt.use_Lora}+viformer:{opt.use_viformer}+lambda:{opt.lamba}')
    print("model's params for training:", count_parameters(clip_model))
    print_training_layers(clip_model)

    model_text = torch.nn.DataParallel(model_text, device_ids=opt.gpu_ids)
    model_image = torch.nn.DataParallel(model_image, device_ids=opt.gpu_ids)

    return model_text, model_image, clip_model, preprocess

def load_pretrain(device, model_name, checkpoint=None, is_ovt_model=False):
    if is_ovt_model:
        clip_model, preprocess = clip.load(model_name,device=device,jit=False)
        convert_models_to_fp32(clip_model)
        # use naive multihead atten to replace nn.multiheadatten in visual
        for module in clip_model.visual.transformer.resblocks:
            new_module = PlainMultiHeadAttention(embed_dim=module.attn.embed_dim, num_heads=module.attn.num_heads)
            new_module.set_parameters(module.attn)
            module.attn = new_module
        # print(clip_model)

        model_text = TextCLIP(clip_model)
        model_image = ImageCLIP(clip_model)

        # print(model_image)
        # print([key for key, _ in model_image.named_parameters()])

        if opt.use_Lora:
            lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, target_modules=['qkv'])
            model_image = get_peft_model(model_image, lora_config)

        if opt.use_viformer:
            feature_dim = clip_model.visual.output_dim
            viformer = VIformer(feature_dim=feature_dim, num_transformer_layers=opt.num_transformer_layers, nhead=opt.nhead, dim_feedforward=2048).to(device)

            clip_model = CLIPWithVIformer(clip_model, viformer).to(device)

        model_text = TextCLIP(clip_model)
        model_image = ImageCLIP(clip_model)
        model_text = torch.nn.DataParallel(model_text, device_ids=opt.gpu_ids)
        model_image = torch.nn.DataParallel(model_image, device_ids=opt.gpu_ids)

        print(f"using OVT-CLIP-{model_name}")
        print(f'loading checkpoint:{checkpoint}')
        checkpoint = torch.load(checkpoint)
        clip_model.load_state_dict(checkpoint['model_state_dict'])

        return clip_model, preprocess

    else:
        clip_model, preprocess = clip.load(model_name,device=device,jit=False)
    
        return clip_model, preprocess