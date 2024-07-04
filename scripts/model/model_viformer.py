import torch
from torch import nn
from torchvision.models import resnet50
from transformers import BertConfig, BertModel
import clip
import numpy as np
import torch.nn.functional as F
from config import get_opts
opt = get_opts()

class MLPFeatureTransform(nn.Module):
    def __init__(self, input_dim):
        super(MLPFeatureTransform, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VIformer(nn.Module):
    def __init__(self, feature_dim, num_transformer_layers=2, nhead=4, dim_feedforward=2048):
        super(VIformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

    def forward(self, x):
        # Transformer expects input of shape (S, N, E)
        # S is the source sequence length, N is the batch size, E is the feature number
        # CLIP features are of shape (N, E), so we add a dummy dimension at position 0
        x = x.unsqueeze(0)
        x_transformed = self.transformer_encoder(x)
        x_transformed = x_transformed.squeeze(0)
        return x_transformed

class CLIPWithVIformer(nn.Module):
    def __init__(self, clip_model, transformer_module):
        super(CLIPWithVIformer, self).__init__()
        self.clip_model = clip_model
        # self.transformer_module = transformer_module.half()
        self.transformer_module = transformer_module
        # Freeze the CLIP model
        if not opt.use_Lora:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, texts=None):
        clip_features = self.clip_model.encode_image(images)
        transformed_image_features = clip_features + self.transformer_module(clip_features)

        if texts is not None:
            text_features = self.clip_model.encode_text(texts)
            # logits_per_image = (transformed_image_features @ text_features.t()).softmax(dim=-1)
            # logits_per_text = (text_features @ transformed_image_features.t()).softmax(dim=-1)

            transformed_image_features = transformed_image_features / transformed_image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * transformed_image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            return logits_per_image, logits_per_text
        else:
            return transformed_image_features

    def encode_image(self, images):
        clip_features = self.clip_model.encode_image(images)
        # We add the original CLIP features (residual connection)
        transformed_features = clip_features + 0.1*self.transformer_module(clip_features)
        transformed_features = F.normalize(transformed_features, p=2, dim=1)
        return transformed_features

    def encode_text(self, texts):
        return self.clip_model.encode_text(texts)
        

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/16",device=device,jit=False)

# feature_dim = clip_model.visual.output_dim
# VIformer = VIformer(feature_dim).to(device)

# # Create the combined model
# model = CLIPWithVIformer(clip_model, VIformer)

# # Example input (batch of images)
# images = torch.randn((3, 3, 224, 224), device=device)
# text = clip.tokenize(['abc','abc','abc']).to(device)
# # Forward pass
# logits_per_image, logits_per_text = model(images, text)
# print(logits_per_image.shape)
# print(logits_per_text.shape)

# output = model.encode_image(images)
# print(output.shape)
