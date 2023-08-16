import os
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


layer_scale = False
init_value = 1e-6

if os.getcwd().split('/')[-1] == 'main':
    pretrained_path = 'pretrain_model/uniformer_base_k400_16x4.pth'
else:
    pretrained_path = 'main/pretrain_model/uniformer_base_k400_16x4.pth'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_visualization=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_visualization:
            attn_copy = attn.clone().detach().cpu().numpy() # [N, num_head, T, T]
            attn_copy = np.sum(attn_copy, axis=2) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_visualization:
            return x, attn_copy
        return x, None

class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # self.global_filter = GlobalFilter(dim=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CBlock_add_G(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.global_filter = GlobalFilter(dim=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.global_filter(self.conv1(self.norm1(x))))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.glb_filter = GlobalFilter(dim=dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x, return_visualization=False):
        x = x + self.pos_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x_attn, visualization_heads = self.drop_path(self.gamma_1 * self.attn(self.norm1(x), return_visualization))
            x = x + x_attn
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x_attn, visualization_heads = self.attn(self.norm1(x), return_visualization)
            x_attn = self.drop_path(x_attn)
            x = x + x_attn
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        if return_visualization:
            return x, visualization_heads
        else:
            return x
   

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

class Conv_adaptive(nn.Module):
    """
    重写第一个卷积层
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=8, embed_dim=64, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, groups=in_chans, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, d=4, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(d, h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.d = d
        self.dim = dim

    def forward(self, x, spatial_size=None):
        B, C, D, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # B, N, C = x.shape
        # # B, C, N, _, _ = x.shape
        # if spatial_size is None:
        #     b = c = int(math.sqrt(N / 4))
        #     # a = b = c = int(N**(1/3))
        #     a = 4
        # else:
        #     a, b, c = spatial_size

        # x = x.view(B, a, b, c, C)
        x = x.to(torch.float32)

        x = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(2, 3, 4), norm='ortho')

        # x = x.reshape(B, N, C)
        # x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class UniFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, return_visualization=False,
                 drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=None, conv_stem=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            # self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            # self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2])
            # self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3])
    
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2))

        else:
            # self.patch_embed1 = PatchEmbed(
            #     img_size=img_size, patch_size=2, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=(1, 2, 2), in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=(1, 2, 2), in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=(1, 2, 2), in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(1, 2, 2))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.return_visualization = return_visualization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # if isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3: # [N, 320, 4, 7, 7] td [N, 196, 320]
            x = blk(x)
        x = self.patch_embed4(x) # [N, 512, 4, 3,3] td [N, 26, 512]
        for i, blk in enumerate(self.blocks4):
            if i < len(self.blocks4) -1:
                x = blk(x)
            else:
                x, visualization_heads = blk(x, return_visualization=self.return_visualization)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x, visualization_heads

    def forward(self, x):
        x, visualization_heads = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x, visualization_heads


def uniformer_small(pretrained=True, return_visualization=False, **kwargs):
    model = UniFormer(
                    depth=[5, 8, 20, 7],
                    embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), return_visualization=return_visualization, **kwargs)
    model.default_cfg = _cfg()
    model_para = torch.load(pretrained_path)
    for para_name in list(model_para.keys()):
        if 'patch_embed1' in para_name:
            model_para.pop(para_name)
        # if 'head' in para_name:
        #     model_para.pop(para_name)

    model.load_state_dict(model_para, strict=True)

    return model

class Merge_uniformer_Base(nn.Module):
    def __init__(self, num_classes=400,
                       num_head = 7,
                       num_phase=8,
                       pretrained=None, 
                       pretrained_cfg=None,
                       return_visualization=False,
                       **kwargs):
        super().__init__()
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model1 = Conv_adaptive(img_size=114, patch_size=(3, 2, 2), in_chans=num_phase, embed_dim=64, stride=None)
        self.model2 = uniformer_small(in_chans=num_phase, num_classes=num_classes, return_visualization=return_visualization, **kwargs)
        self.model2.default_cfg = _cfg()
        # self.linear = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 7)
        # )
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 7))
        # self.linear = nn.Linear(512, 7)
        self.return_visualization = return_visualization

    def forward(self, x):
        # x = x[:, 0, :]
        # N, Z, H, W = x.shape
        # x = x.unsqueeze(1)
        # N, T, Z, H, W = x.shape
        x = self.model1(x)
        x, visualization_heads = self.model2.forward_features(x)
        # x, visualization_heads = self.model2(x)
        # if self.return_visualization:
        #     depth, height, width = 4, 3, 3
        #     visualization_heads = visualization_heads.reshape((visualization_heads.shape[0],visualization_heads.shape[1],depth, height, width))
        x = x.flatten(2).mean(-1)
        x = self.dropout(self.linear(x))


        return x, visualization_heads