# 2023.09.18-Changed for transformer of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# This file editing from https://github.com/hustvl/TopFormer/blob/main/mmseg/models/backbones/topformer.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from .layers import Conv2d_BN, DropPath, h_sigmoid
# from einops import rearrange


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape
class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c):
        """
        :param
        in_c: 进行注意力refine的特征图的通道数目；
        原文中的降维和升维没有使用
        """
        super(DoubleAtten,self).__init__()
        self.in_c = in_c
        """
        以下对同一输入特征图进行卷积，产生三个尺度相同的特征图，即为文中提到A, B, V
        """
        self.convA = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c,kernel_size=1)
    def forward(self,input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w) # 对 A 进行reshape
        atten_map = atten_map.view(b, self.in_c, 1, h*w)       # 对 B 进行reshape 生成 attention_aps
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # 特征图与attention_maps 相乘生成全局特征描述子

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1) # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1) # 注意力向量左乘全局特征描述子

        return out.view(b, _, h, w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention2(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), focusing_factor=3, kernel_size=5
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim) # 实验中为2 
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.dim = dim
        self.focusing_factor = focusing_factor

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.to_flattenq = nn.Linear(dim, nh_kd, True)
        self.to_flattenk = nn.Linear(dim, nh_kd, True)
        self.to_flattenv = nn.Linear(dim, self.dh, True)

        self.to_testV = nn.Linear(dim, self.dh, True)

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, key_dim)))
        self.dwc = nn.Conv2d(in_channels=self.num_heads, out_channels=self.num_heads, kernel_size=kernel_size,
                             groups=self.num_heads, padding=kernel_size // 2)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        
        B, C, H, W = get_shape(x)
        # print('----------------------------')
        # print(B, C, H, W)
        # print('----------------------------')
        # print(get_shape(x))
        # print('--------------------------------------------------------')
        # print('heads:', self.num_heads)
        # print('key:', self.key_dim)
        # print('attn_ratio:', self.attn_ratio)
        # print('d:',self.d)
        # print('nh_kd:',self.nh_kd)
        # print('dim',self.dim)
        # print('dh',self.dh)


        # qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        # kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        # vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        
  
        # print('qq:',get_shape(qq))
        # print('kk:',get_shape(kk))
        # print('vv:',get_shape(vv))



        # print('-----------------------------test---------------------------')
        # test = self.to_test(x.view(B, -1, self.dim)).view(B, self.nh_kd, H, W).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        # print('test:',get_shape(test))
        # p = 3 
        # kernel_function = nn.ReLU()
        # test = kernel_function(test) + 1e-6 
        # print(get_shape(self.scale))
        # scale = nn.Softplus()(self.scale)
        # print(get_shape(scale))
        # test = test/scale
        # test_norm = test.norm(dim = -1, keepdim = True)
        # test = test ** p 
        # test = (test/test.norm(dim = -1, keepdim = True))*test_norm
        # print('test:',get_shape(test))
        # print('-----------------------------test---------------------------')


        # print('-----------------------------testV---------------------------')
        # testvv = self.to_testV(x.view(B, -1, self.dim)).view(B, self.dh, H, W)
        # print('testvv:',get_shape(testvv))
        # testvv = testvv.reshape(B, self.num_heads, self.d, H * W)
        # print('testvv:',get_shape(testvv))
        # testvv = testvv.permute(0, 1, 3, 2)
        # print('testvv:',get_shape(testvv))

        # feature_map = self.dwc(testvv)
        # print('feature_map:',get_shape(feature_map))
        # # test = test + feature_map
        # # print('test:',get_shape(test))
        # print('-----------------------------testV---------------------------')


        # print('-----------------------------ALL---------------------------')
        fq = self.to_flattenq(x.view(B, -1, self.dim)).view(B, self.nh_kd, H, W).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        fk = self.to_flattenk(x.view(B, -1, self.dim)).view(B, self.nh_kd, H, W).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        fv = self.to_testV(x.view(B, -1, self.dim)).view(B, self.dh, H, W).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        fq = kernel_function(fq) + 1e-6
        fk = kernel_function(fk) + 1e-6
        scale = nn.Softplus()(self.scale)
        fq = fq / scale
        fk = fk / scale
        fq_norm = fq.norm(dim = -1, keepdim = True)
        fk_norm = fk.norm(dim = -1, keepdim = True)
        fq = fq ** focusing_factor
        fk = fk ** focusing_factor
        fq = (fq / fq.norm(dim = -1, keepdim = True))*fq_norm
        fk = (fk / fk.norm(dim = -1, keepdim = True))*fk_norm
        fk = fk.reshape(B, self.num_heads, self.key_dim, H * W) 
        feature_map = self.dwc(fv)
        # test = test + feature_map




        attn = torch.matmul(fq, fk)
        # print('attn',get_shape(attn)) # torch.Size([1, 4, 100, 100])
        attn = attn.softmax(dim=-1)  # dim = k
        # print('attn',get_shape(attn)) # torch.Size([1, 4, 100, 100])
        xx = torch.matmul(attn, fv)
        # print('xx',get_shape(xx))  #([1, 4, 16, 100]) 
        xx = xx + feature_map

        # xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = xx.permute(0,1,3,2)
        # print('xx',get_shape(xx))
        xx = xx.reshape(B, self.dh, H, W)

        # print('xx',get_shape(xx))
        xx = self.proj(xx)
        return xx



class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx



class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True),focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        #self.attn = Attention2(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
         #              norm_cfg=norm_cfg)


        #self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
         #                     norm_cfg=norm_cfg)

        self.attn = DoubleAtten(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_cfg=norm_cfg, act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(
        np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return torch.cat(out, dim=1)


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg

        if not global_inp:
            global_inp = inp

        self.local_embedding = ConvModule(
            inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(
            global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(
            global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(
                H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(
                H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out



