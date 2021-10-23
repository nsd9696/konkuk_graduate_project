""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.functional import Tensor
import torch.nn as nn
from functools import partial
import math
import warnings




class Ffn(nn.Module):
    # feed forward network layer after attention
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        N, L, D = q.shape
        q, k, v = self.query(q), self.key(k), self.value(v)
        q = q.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos):
        x = self.norm1(x)
        q, k, v = x + pos, x + pos, x
        x = x + self.attn(q, k, v)
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn2 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos, task_embed):
        memory = x
        x = self.norm1(x)
        q, k, v = x + task_embed, x + task_embed, x
        x = x + self.attn1(q, k, v)
        x = self.norm2(x)
        q, k, v = x + task_embed, memory + pos, memory
        x = x + self.attn2(q, k, v)
        x = x + self.ffn(self.norm3(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                     padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                     padding=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += residual
        # out = self.relu(out)

        return out

class Head(nn.Module):
    """ Head consisting of convolution layers
    Extract features from corrupted images, mapping N3HW images into NCHW feature map.
    """
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
        # self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.resblock1(out)
        out = self.resblock2(out)

        return out

class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    input : N C H W
    output: N num_patch P^2*C
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x):
        N, C, H, W = ori_shape = x.shape
        
        p = self.patch_size
        num_patches = (H // p) * (W // p)
        out = torch.zeros((N, num_patches, self.dim)).to(x.device)
        #print(f"feature map size: {ori_shape}, embedding size: {out.shape}")
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out, ori_shape

class DePatchEmbed(nn.Module):
    """ Patch Embedding to Feature
    input : N num_patch P^2*C
    output: N C H W
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, ori_shape):
        N, num_patches, dim = x.shape
        _, C, H, W = ori_shape
        p = self.patch_size
        out = torch.zeros(ori_shape).to(x.device)
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, :, i:i+p, j:j+p] = x[:, k, :].reshape(N, C, p, p)
            #out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out


class Tail(nn.Module):
    """ Tail consisting of convolution layers and pixel shuffle layers
    NCHW -> N3HW.
    """
    def __init__(self, task_id, in_channels, out_channels):
        super(Tail, self).__init__()
        assert 0 <= task_id <= 5
        # 0, 1 for noise 30, 50; 2, 3, 4 for sr x2, x3, x4, 5 for defog
        upscale_map = [1, 1, 2, 3, 4, 1]
        scale = upscale_map[task_id]
        m = []
        # for SR task
        if scale > 1:
            m.append(nn.Conv2d(in_channels, in_channels * scale * scale, kernel_size=3, stride=1,
                     padding=1, bias=False))
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    m.append(nn.PixelShuffle(2))
            elif scale == 3:
                m.append(nn.PixelShuffle(3))
            else:
                raise NameError("Only support x3 and x2^n SR")

        m.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False))
        self.m = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.m(x)
        #print("task_id:", self.task_id)
        #print("shape of tail's output:", x.shape)
        # out = self.bn1(out)
        return out