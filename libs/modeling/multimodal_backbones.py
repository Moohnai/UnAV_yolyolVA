from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from .models import register_multimodal_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock,  MaskedMHCA,
                    MaskedConv1D, LayerNorm)
from mmengine.model import BaseModule
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
# from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
 


        ########m: define downsampling and normalization class
class Downsample_pyramid_levels(nn.Module):
    def __init__(self, n_embd, scale_factor):

        super().__init__()

        self.n_embd = n_embd

        assert (scale_factor == 1) or (scale_factor % 2 == 0)
        self.x_stride = scale_factor      # dowsampling stride for input

        kernel_size = self.x_stride + 1 if self.x_stride > 1 else 3
        stride, padding = self.x_stride, kernel_size // 2


        # downsample conv
        self.down_conv = MaskedConv1D(
        self.n_embd, self.n_embd, kernel_size=3,
        stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.down_norm = LayerNorm(self.n_embd)

    def forward(self, x, mask):
        x, mask = self.down_conv(x, mask) # [B, T, C]
        x = self.down_norm(x) # [B, T, C]

        return x, mask


class CSPLayerWithTwoConv(BaseModule):
    """Cross Stage Partial Layer with 2 convolutions.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg = None,
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg = dict(type='SiLU', inplace=True),
            init_cfg = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = MaskedConv1D(
            in_channels,
            2 * self.mid_channels,
            1,
            )
        self.final_conv = MaskedConv1D(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            )

        self.blocks = nn.ModuleList(
            MaskedMHCA(
                    self.mid_channels,
                    n_head=4,
                    n_qx_stride=1,
                    n_kv_stride=1,
                    attn_pdrop=0,
                    proj_pdrop=0
                    ) for _ in range(num_blocks))

    def forward(self, x: Tensor, mask) -> Tensor:                                              #top-down                       bottom-up 
        """Forward process."""                                                         #x: [1, 1280, 40, 40]        x: [1, 960, 40, 40]
        x_main = self.main_conv(x)                                                 #[1,640,40,40]                       [1,640,40,40]
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))      #[1,320,40,40],[1,320,40,40]       [1,320,40,40],[1,320,40,40]
        for blocks in self.blocks:
            x_ , mask = blocks(x_main[-1], mask)
            x_main.append(x_)
        # x_main.extend(blocks(x_main[-1], mask)[0] for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1)), mask                                #[1,640,40,40]                      [1,640,40,40]

class MaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg = None,
                 norm_cfg = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = MaskedConv1D

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = MaskedConv1D(
            in_channels,
            embed_channels,
            1,
            ) if embed_channels != in_channels else None
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 )

    def forward(self, x: Tensor, guide: Tensor, mask) -> Tensor:
        """Forward process."""
        B, _, H = x.shape #[B, C_level, H, W]
        #guid: [B, n_cls, n_feat]
        guide = self.guide_fc(guide) #[B, n_cls, C_level]
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed, mask = self.embed_conv(x, mask) if self.embed_conv is not None else x, mask #[B, C_level, H, W]:x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H)

        if self.use_einsum:
            attn_weight = torch.einsum('bmch,bnmc->bmhn', embed, guide) #[B, num_head, H, W, n_cls] [8,8,112,512]/ embed:[8,8,32,112]/guide:[8,512,8,32]
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            # prevent attn from attending to invalid tokens
            attn_weight = attn_weight.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0] #max over the classes #[8,8,112]
        attn_weight = attn_weight / (self.head_channels**0.5) #normalization in attention #[8,8,112]
        attn_weight = attn_weight + self.bias[None, :, None] #[8,8,112]
        attn_weight = attn_weight.sigmoid() * self.scale #[8,8,112]

        x, mask = self.project_conv(x, mask) #same  #[8,256,112],[8,1,112]
        x = x.reshape(B, self.num_heads, -1, H)  #[8,8,32,112]
        x = x * attn_weight.unsqueeze(2)        #[8,8,32,112]
        x = x.reshape(B, -1, H)
        return x, mask

class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int, # 1280
            out_channels: int, # 640
            guide_channels: int, # 512
            embed_channels: int, # 320
            num_heads: int = 1, # 10
            expand_ratio: float = 0.5,
            num_blocks: int = 1, # 3
            with_scale: bool = False,
            add_identity: bool = False,  # shortcut
            conv_cfg = None,
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg = dict(type='SiLU', inplace=True),
            init_cfg = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = MaskedConv1D((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     )

        self.attn_block = MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg,
                                              use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor, mask) -> Tensor:
        """Forward process."""
        x_main, mask = self.main_conv(x,mask) # x: [1, 1280, 40,40] , x_main: [1, 640, 40, 40]
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1)) # [1, 320, 40, 40], [1, 320, 40, 40]

        for blocks in self.blocks:
            x_ , mask = blocks(x_main[-1], x_main[-1], mask)
            x_main.append(x_)
        # x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)  # [1, 320, 40, 40], [1, 320, 40, 40] --> 5 members
        x_, mask = self.attn_block(x_main[-1], guide, mask)
        x_main.append(x_)
        # x_main.append(self.attn_block(x_main[-1], guide))
        x_main, mask = self.final_conv(torch.cat(x_main, 1), mask)
        return x_main, mask
    

class ImagePoolingAttentionModule(nn.Module):

    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3,
                 use_einsum: bool = True):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        self.use_einsum = use_einsum
        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   nn.Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 nn.Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   nn.Linear(embed_channels, embed_channels))
        self.proj = nn.Linear(embed_channels, text_channels)

        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])

    def forward(self, text_features, image_features):
        B = image_features[0].shape[0] #3 diffrent feature levels [B, n_feat, H, W]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2) #[B,_, H, W]--->[B,_,num_patches]
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        if self.use_einsum:
            attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(q, k)

        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        if self.use_einsum:
            x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        else:
            v = v.permute(0, 2, 1, 3)
            x = torch.matmul(attn_weight, v)
            x = x.permute(0, 2, 1, 3)
        x = self.proj(x.reshape(B, -1, self.embed_channels))
        return x * self.scale + text_features

class downsample(nn.Module):
    def __init__(self, n_embd, scale_factor=2):
        super().__init__()
        self.n_embd = n_embd
        self.scale_factor = scale_factor
        assert (scale_factor == 1) or (scale_factor % 2 == 0)
        self.x_stride = scale_factor
        kernel_size = self.x_stride + 1 if self.x_stride > 1 else 3
        stride, padding = self.x_stride, kernel_size // 2
        self.down_conv = MaskedConv1D(
                    self.n_embd, self.n_embd, kernel_size=kernel_size,
                    stride=stride, padding=padding
                )
        self.down_norm = LayerNorm(self.n_embd)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.down_conv(x, mask)
        x = self.down_norm(x)
        x = self.act(x)
        return x, mask

class MaskedAdaptiveMaxPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, mask):
        x = super().forward(x)
        mask = F.interpolate(mask.float(), size=x.size(-1), mode='nearest')
        mask = mask > 0.5
        return x, mask
    
class fusion_module(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.scale_factor = 2
        # self.in_channels = [224, 112, 56]
        self.in_channels = [224, 112, 56, 28, 14, 7]
        self.upsample_feats_cat_first = True

        self.reduce_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.reduce_layers.append(nn.Identity())
        
        # self.text_enhancer = ImagePoolingAttentionModule(..)
        self.text_enhancer = MaskedMHCA(
                        n_embd,
                        n_head=4,
                        n_qx_stride=1,
                        n_kv_stride=1,
                        attn_pdrop=0,
                        proj_pdrop=0
                        )

        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(nn.Upsample(scale_factor=self.scale_factor, mode='nearest'))
        self.upsample_layers.append(nn.Upsample(scale_factor=self.scale_factor, mode='nearest'))
        self.upsample_layers.append(nn.Upsample(scale_factor=self.scale_factor, mode='nearest'))
        self.upsample_layers.append(nn.Upsample(scale_factor=self.scale_factor, mode='nearest'))
        self.upsample_layers.append(nn.Upsample(scale_factor=self.scale_factor, mode='nearest'))
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(downsample(n_embd, scale_factor=self.scale_factor))
        self.downsample_layers.append(downsample(n_embd, scale_factor=self.scale_factor))
        self.downsample_layers.append(downsample(n_embd, scale_factor=self.scale_factor))
        self.downsample_layers.append(downsample(n_embd, scale_factor=self.scale_factor))
        self.downsample_layers.append(downsample(n_embd, scale_factor=self.scale_factor))

        self.top_down_layers = nn.ModuleList()
        self.top_down_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))
        self.top_down_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=4 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.top_down_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=4 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.top_down_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=4 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.top_down_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=4 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.bottom_up_layers = nn.ModuleList()
        self.bottom_up_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))
        self.bottom_up_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))
        self.bottom_up_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.bottom_up_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))

        self.bottom_up_layers.append(MaxSigmoidCSPLayerWithTwoConv(
            in_channels=1024,# 1280
            out_channels=512, # 640
            guide_channels=224, # 512
            embed_channels=256, # 320
            num_heads=8 , # 10
            expand_ratio= 0.5,
            num_blocks=3, # 3
        ))
        
        self.out_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.out_layers.append(nn.Identity())

        ######
        embed_channels = 512
        self.projections = nn.ModuleList([
            MaskedConv1D(in_channels, embed_channels, 1,)
            for in_channels in [512, 512, 512]
        ])
        
        self.pool_size = 4
        self.num_feats = 3
        self.image_pools = nn.ModuleList([
            MaskedAdaptiveMaxPool1d((self.pool_size))
            for _ in range(self.num_feats)
        ])

        self.match_projection = nn.Conv1d(3*self.pool_size, 224, 1,)

    
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor, mask_img, mask_txt) -> tuple:
        """Forward function."""
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx])) # [B, C_level, 224], [B, 512, 112], [B, 512, 56], [B, 512, 28], [B, 512, 14], [B, 512, 7]

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0] #[B, 512, 56]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            ## apply mask for upsampled features -- repeat ones and zeros for the mask by scale factor
            mask_img_up = mask_img[idx].repeat_interleave(self.scale_factor, dim=-1)
            mask_img_up = F.interpolate(mask_img_up.float(), size=upsample_feat.shape[-1], mode='nearest')
            mask_img_up = mask_img_up > 0.5


            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)

            inner_out, mask_img_up = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats, mask_img_up)
            
            inner_outs.insert(0, inner_out)  #inner_outs:[8,512,224], [8,512,112], [8,512,56]


        B = inner_outs[0].shape[0] #3 diffrent feature levels [B, n_feat, H, W] #[1,320,80,80]
        num_patches = self.pool_size**2
        # mlvl_image_features = [             #[1,256,9],[1,256,9],[1,256,9]
        #     pool(proj(x, mask)[0], mask)[0].view(B, -1, num_patches)
        #     for (x, proj, pool, mask
        #          ) in zip(inner_outs, self.projections, self.image_pools, mask_img)
        # ]    
        mlvl_image_features = [             #[1,256,9],[1,256,9],[1,256,9]
            pool(x, mask)[0]
            for (x, proj, pool, mask
                 ) in zip(inner_outs, self.projections, self.image_pools, mask_img)
        ]                                                   
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        mlvl_image_features = self.match_projection(mlvl_image_features).transpose(1, 2)
        
        txt_feats, mask_txt = self.text_enhancer(txt_feats, mlvl_image_features, mask_txt) #visual:[512,224]
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat, mask_img_down = self.downsample_layers[idx](feat_low, mask_img[idx])

            out, mask_img_down = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats, mask_img_down)
            
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results), txt_feats, mask_img, mask_txt


        ############
    
 
@register_multimodal_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in_V,                # input visual feature dimension
        n_in_A,                # input audio feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]-1): 
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for idx in range(arch[2]):
            self.cross_att_Va.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.cross_att_Av.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )


        ######m: add downsample class ######
        self.downsample_list = nn.ModuleList()
        for idx in range(5): #self.downsample_list
            self.downsample_list.append(Downsample_pyramid_levels(n_embd, scale_factor))

        ######m: add fusion module ######
        self.fusion_module = fusion_module(n_embd)

        ####################################

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C_V, T = x_V.size()
        mask_V = mask_A = mask
        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # stem transformer
        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

        

            
        ########### adding yolo-world fusion module with down-sampling ###########
        x_V_org = x_V
        mask_V_org = mask_V
        x_V_list = [x_V]        #[B, 512, 224], [B, 512, 112], [B, 512, 56]
        mask_V_list = [mask_V]  #[B, 1, 224], [B, 1, 112], [B, 1, 56]
        for idx in range(len(self.downsample_list)):
            x_V, mask_V = self.downsample_list[idx](x_V_list[-1], mask_V_list[-1])
            x_V_list.append(x_V) ###m: list of features of a batch for 3 levels. [8,512,224], [8,512,112], [8,512,56]
            mask_V_list.append(mask_V) ###m: list of maskes of a batch for 3 levels. [8,1,224], [8,1,112], [8,1,56]
        x_V_fusion, x_A_fusion, mask_V_fusion, mask_A_fusion = self.fusion_module(x_V_list, x_A, mask_V_list, mask_A)

        out_feats_V = x_V_fusion
        out_masks_V = tuple(mask_V_fusion)
        ###############adding yolo-world fusion module with down-sampling for audio############
        x_A_list = [x_A]        
        mask_A_list = [mask_A]
        for idx in range(len(self.downsample_list)):
            x_A, mask_A = self.downsample_list[idx](x_A_list[-1], mask_A_list[-1])
            x_A_list.append(x_A) 
            mask_A_list.append(mask_A)
        x_A_fusion, x_V_fusion, mask_A_fusion, mask_V_fusion = self.fusion_module(x_A_list, x_V_org, mask_A_list, mask_V_org) #[B,512,224]

        out_feats_A = x_A_fusion
        out_masks_A = tuple(mask_A_fusion)

        # out_feats_A = [x_A_fusion]
        # for idx in range(5):
        #     out_A, mask_A_fusion = self.downsample_list[0](out_feats_A[-1], mask_A_fusion)
        #     out_feats_A.append(out_A)
        # out_feats_A = tuple(out_feats_A)

        return out_feats_V, out_feats_A, out_masks_V