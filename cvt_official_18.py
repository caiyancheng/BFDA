from functools import partial
from itertools import repeat
from torch._six import container_abcs
import numpy as np
import torchvision

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# from net.wrn import WideResNet

from timm.models.layers import DropPath, trunc_normal_

# from .registry import register_model
from show import *

_model_entrypoints = {}


class ConvBNReLU(nn.Module):  # CBR块

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):  # 256,256
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,  # //4
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,  # //4
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)  # [1,28,4,4]
        feat = self.convblk(fcat)  # [1,4,4,4]
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)  # [1,4,1,1]
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def register_model(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]

    _model_entrypoints[model_name] = fn

    return fn


def model_entrypoints(model_name):
    return _model_entrypoints[model_name]


def is_model(model_name):
    return model_name in _model_entrypoints


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        A = 0
        if type(x) == list:
            A = 1
            x, back_attention = x
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        if A:
            attn = (back_attention.unsqueeze(1))*attn


        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        # 直接使用identity阵
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        wether_change_attention = False
        if type(x) == list:
            wether_change_attention = True
            x, back_attention = x
        res = x

        x = self.norm1(x)
        if wether_change_attention:
            attn = self.attn([x,back_attention], h, w)
            x = self.drop_path(attn)
            x = self.drop_path(self.mlp(self.norm2(x)))
        else:
            attn = self.attn(x, h, w)
            x = res + self.drop_path(attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
            # nn.init.normal_(self.cls_token,std=0.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            # nn.init.normal_(m.weight,std=0.02)

            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        wether_first = False
        SHOW = False
        if type(x) == list:
            if len(x) == 4:
                SHOW = True
                x, back_attention, id, save_dir = x
            else:
                x , back_attention = x
            wether_first = True
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if wether_first:
                x = blk([x,back_attention],H,W)
            else:
                x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        if wether_first and SHOW:
            img_path = '/remote-home/share/42/cyc19307140030/yolov5/runs/trainforpaper_crossdomain_transformer/exp37/src_img_aachen_000031_000019.png'
            img = cv2.imread(img_path)
            img_t = toTensor(img)
            show_feature_cam(feature_map=x.clone().cpu(),id=id,img=img_t,save_dir=save_dir)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=10,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = spec['NUM_CLASSES']

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head

        # 修改此部分
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        if num_classes > 0:
            trunc_normal_(self.head.weight, std=0.02)

        # if num_classes > 0:
        #     nn.init.xavier_uniform_(self.head.weight)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            # print('=> loading pretrained model {pretrained}')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] == '*'
                )
                if need_init:
                    # print('need_init',need_init)
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                                .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                                .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            # print(need_init_state_dict)
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        SHOW = False
        List = False
        if type(x) == list:
            List = True
            if len(x) == 4:
                SHOW = True
            W = 0
            if SHOW:
                x, back_attention, id, save_dir =x
            else:
                x, back_attention = x
            n_batch = x.shape[0]  #batch_size
            if type(back_attention) == list:#意味着有两层
                W = 1
                back_attention, back_attention_2 = back_attention
        else:
            n_batch = x.shape[0]
        for i in range(self.num_stages): #3
            if List:#只在第一层的attention层加上该back——mask
                if i == 0:
                    if SHOW == True:
                        x, cls_tokens = getattr(self, f'stage{i}')([x, back_attention, id, save_dir])
                    else:
                        x, cls_tokens = getattr(self, f'stage{i}')([x,back_attention])
                if W == 1:
                    if i == 1:
                        x, cls_tokens = getattr(self, f'stage{i}')([x, back_attention_2])
                if W == 0:
                    if i == 1:
                        x, cls_tokens = getattr(self, f'stage{i}')(x)
                if i != 0 and i != 1:
                    x, cls_tokens = getattr(self, f'stage{i}')(x)
            else:
                x, cls_tokens = getattr(self, f'stage{i}')(x)
        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
            x = x.view(n_batch, -1)
        else:
            # print('not cls token',x.shape)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        if type(x) == list:
            if len(x) == 2:
                x, back_attention = x
                x = self.forward_features([x, back_attention])
                x = self.head(x)
            elif len(x) == 4:
                x, back_attention, id, save_dir = x
                x = self.forward_features([x, back_attention, id, save_dir])
                x = self.head(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)

        return x


@register_model
def get_cls_model(config, **kwargs):
    msvit_spec = config.MODEL.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=10,
        act_layer=QuickGELU,
        norm_layer=nn.LayerNorm,
        init='trunc_norm',
        spec=msvit_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        msvit.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return msvit


CVT13_SPEC = {
    'NUM_STAGES': 3,
    'NUM_CLASSES': 10,
    'PATCH_SIZE': [7, 3, 3],
    'PATCH_STRIDE': [4, 2, 2],
    'PATCH_PADDING': [0, 0, 0],
    'DIM_EMBED': [64, 192, 384],
    'DEPTH': [1, 2, 10],
    'NUM_HEADS': [1, 3, 6],
    'MLP_RATIO': [4, 4, 4],
    'QKV_BIAS': [False, False, False],
    'DROP_RATE': [0, 0, 0],
    'ATTN_DROP_RATE': [0, 0, 0],
    'DROP_PATH_RATE': [0, 0, 0],
    'CLS_TOKEN': [False, False, True],
    'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
    'KERNEL_QKV': [3, 3, 3],
    'PADDING_Q': [1, 1, 1],
    "PADDING_KV": [1, 1, 1],
    'STRIDE_KV': [1, 1, 1],
    'STRIDE_Q': [1, 1, 1],
}


def get_cvt13_pretrained(pth):
    cvt = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=10,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init='trunc_norm',
        spec=CVT13_SPEC)
    cvt.init_weights(
        pth,
        ['stage0', 'stage1', 'stage2'],
        True, )
    return cvt


class Base_cvt_cifar(nn.Module):
    def __init__(self, pth):
        super().__init__()
        # self.up = F.interpolate(size=(224, 224), mode='bilinear', align_corners=True)
        # self.up = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        self.cvt = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=2,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            init='trunc_norm',
            spec=CVT13_SPEC)
        self.cvt.init_weights(pth, ['stage0', 'stage1', 'stage2'], True)
        self.change_channel = nn.Conv2d(18,3, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        if type(x) == list:
            if len(x) == 2:
                x, back_attention = x
                x = F.interpolate(x,size=(224, 224), mode='bilinear', align_corners=True)
                x = self.change_channel(x)
                # x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)
                x = self.cvt([x, back_attention])
            elif len(x) == 4:
                x, back_attention, id, save_dir = x
                x = F.interpolate(x,size=(224, 224), mode='bilinear', align_corners=True)
                x = self.change_channel(x)
                # x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)
                x = self.cvt(x)
        else:
            x = F.interpolate(x,size=(224, 224), mode='bilinear', align_corners=True)
            x = self.change_channel(x)
            # x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)
            x = self.cvt(x)

        return x

if __name__ == '__main__':
    # cvt = ConvolutionalVisionTransformer(
    #     in_chans=3,
    #     num_classes=2,
    #     act_layer=nn.GELU,
    #     norm_layer=nn.LayerNorm,
    #     init='trunc_norm',
    #     spec=CVT13_SPEC)
    # cvt.init_weights(
    #     'CvT-13-224x224-IN-1k.pth',
    #     ['stage0', 'stage1', 'stage2'],
    #     True, )
    #
    # inputs = torch.randn(2, 3, 224, 224)
    # back1 = torch.randn(2,3025,3025)
    # # back2 = torch.randn(2,729,729)
    # outputs = cvt([inputs,back1])
    # # outputs = cvt([inputs, back1, 0, 'ruo'])
    inputs = torch.randn(2, 32, 224, 224)
    pth = 'CvT-13-224x224-IN-1k.pth'
    md = Base_cvt_cifar(pth)
    outputs = md(inputs)
    print(outputs)