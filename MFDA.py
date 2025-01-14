import os
import numpy as np
import torch
from torch import nn, Tensor
import torch._utils
import torch.nn.functional as F
import torchvision.models as models 
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

BN_MOMENTUM = 0.01
from segmentation_models_pytorch.base import modules as md
from torch.autograd import Variable, Function



# from segmentation_models_pytorch

class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            md.Conv2dReLU(
                in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm
            ),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_bathcnorm=use_bathcnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
        self, encoder_channels, use_batchnorm=True, out_channels=512, dropout=0.2
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = md.Conv2dReLU(
            in_channels=encoder_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, features):
        # features up 8x
        x = features
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, dpr=0.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, dpr=0.) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SABlock_Windows(nn.Module):
    def __init__(self, dim, num_heads, window_size=14, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size=window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, int(dim*4))
        # self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2).reshape(B, C, H, W)
        return x 

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class UniFormer(nn.Module):     
    def __init__(self, input_num = 14,model_name: str = 'S') -> None:
        super().__init__()
        depth = [2,2,6,2]
        self.window_size = 4
        head_dim = 16
        drop_path_rate = 0.
        embed_dims = [32, 64, 128, 256]
        self.input_num = input_num
    
        for i in range(4):
            self.add_module(f"patch_embed{i+1}", PatchEmbed(4 if i == 0 else 2, input_num if i == 0 else embed_dims[i-1], embed_dims[i]))
            self.add_module(f"norm{i+1}", nn.LayerNorm(embed_dims[i]))

        self.pos_drop = nn.Dropout(0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dims]

        self.blocks1 = nn.ModuleList([
            CBlock(embed_dims[0], dpr[i])
        for i in range(depth[0])])
        
        self.blocks2 = nn.ModuleList([
            CBlock(embed_dims[1], dpr[i+depth[0]])
        for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
            SABlock_Windows(dim=embed_dims[2], num_heads=num_heads[2], window_size=self.window_size, qkv_bias=True,mlp_ratio=4.0,drop_path=dpr[i+depth[0]+depth[1]])
        for i in range(depth[2])])

        self.blocks4 = nn.ModuleList([
            SABlock_Windows(dim=embed_dims[3], num_heads=num_heads[3], window_size=self.window_size, qkv_bias=True,mlp_ratio=4.0,drop_path=dpr[i+depth[0]+depth[1]+depth[2]]
                    )
        for i in range(depth[3])])

    def forward(self, x: torch.Tensor):
        outs = []
        return outs

# 继承UniFormer的blocks和参数
class MFDA_Base(UniFormer):
    def __init__(self, name = 'MFDA'):
        super(MFDA_Base, self).__init__(input_num=14,model_name='2262')
        self.window_size = 4
        head_dim = 16
        drop_path_rate = 0.
        self.embed_dims = [32, 64, 128, 256]
        embed_dims = [32, 64, 128, 256]
        self.input_sar = 2
        self.input_amsr = 12
        depth_amsr = [2,2,6,2]
        depth = [2,2,6,2]
        self.pos_drop = nn.Dropout(0.1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dims]

        self.blocks_amsr_1 = nn.ModuleList([
            CBlock(embed_dims[0], dpr[i])
        for i in range(depth_amsr[0])])
        
        self.blocks_amsr_2 = nn.ModuleList([
            CBlock(embed_dims[1], dpr[i+depth[0]])
        for i in range(depth_amsr[1])])

        self.blocks_amsr_3 = nn.ModuleList([
            SABlock_Windows(dim=embed_dims[2], num_heads=num_heads[2], window_size=self.window_size, qkv_bias=True,mlp_ratio=4.0,drop_path=dpr[i+depth[0]+depth[1]])
        for i in range(depth_amsr[2])])
        self.blocks_amsr_4 = nn.ModuleList([
            SABlock_Windows(dim=embed_dims[3], num_heads=num_heads[3], window_size=self.window_size, qkv_bias=True,mlp_ratio=4.0,drop_path=dpr[i+depth[0]+depth[1]+depth[2]]
                    )
        for i in range(depth_amsr[3])])

        for i in range(4):
            self.add_module(f"patch_embed{i+1}", PatchEmbed(4 if i == 0 else 2, 2 if i == 0 else embed_dims[i-1], embed_dims[i]))
            self.add_module(f"norm{i+1}", nn.LayerNorm(embed_dims[i]))

        for i in range(4):
            self.add_module(f"patch_amsr_embed{i+1}", PatchEmbed(1 if i == 0 else 2, 12 if i == 0 else embed_dims[i-1], embed_dims[i]))
            self.add_module(f"norm_amsr_{i+1}", nn.LayerNorm(embed_dims[i]))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=480,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
        )

        self.sic = SegmentationHead(
            in_channels=64,
            out_channels=11,
            kernel_size=3,
            upsampling=4
        )
        self.sod = SegmentationHead(
            in_channels=64,
            out_channels=6,
            kernel_size=3,
            upsampling=4
        )

        self.conv13 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.offsets3 = nn.Conv2d(16, 18, kernel_size=3, padding=1)
        self.DF3 = DeformConv2D(16, 16, kernel_size=3, padding=1)

        self.floe = SegmentationHead(
            in_channels=16,
            out_channels=7,
            kernel_size=3,
            upsampling=4
        )        
        self.tanh = nn.Tanh()
        
        # make the water_mask_decoder
        self.water_decoder = PSPDecoder(64,True,256)
        self.water_head = SegmentationHead(
            in_channels=256,
            out_channels=2,
            kernel_size=3,
            upsampling=8
        )        

    def forward(self,input,D=None,domain = 'source'):
        a = input[:,:2,:,:]
        b = input[:,2:,:,:]
        b = F.interpolate(b, size=(64,64), mode='bilinear', align_corners=True)
        
        encoder_list = []
        a = self.patch_embed1(a)
        a = self.pos_drop(a)
        b = self.patch_amsr_embed1(b)
        b = self.pos_drop(b)
        # a = a + b

        for blk in self.blocks1:
            a = blk(a)
        for blk in self.blocks_amsr_1:
            b = blk(b)
        a = a + b
        out_a= self.norm1(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed2(a)
        b = self.patch_amsr_embed2(b)
        for blk in self.blocks2:
            a = blk(a)
        for blk in self.blocks_amsr_2:
            b = blk(b)
        a = a + b
        out_a= self.norm2(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed3(a)
        b = self.patch_amsr_embed3(b)
        for blk in self.blocks3:
            a = blk(a)
        for blk in self.blocks_amsr_3:
            b = blk(b)
        a = a + b
        out_a= self.norm3(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed4(a)
        b = self.patch_amsr_embed4(b)
        for blk in self.blocks4:
            a = blk(a)
        for blk in self.blocks_amsr_4:
            b = blk(b)
        a = a + b
        out_a= self.norm4(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))
        
        

        x1 = F.interpolate(encoder_list[0], size=(64, 64), mode='bilinear', align_corners=True)
        x2 = F.interpolate(encoder_list[1], size=(64, 64), mode='bilinear', align_corners=True)
        x3 = F.interpolate(encoder_list[2], size=(64, 64), mode='bilinear', align_corners=True)
        x4 = F.interpolate(encoder_list[3], size=(64, 64), mode='bilinear', align_corners=True)
        x = torch.cat([x1,x2,x3,x4],dim=1)

        

        if domain == 'source':
            xt = x
        if domain == 'target':
            aa = D[0](x)
            aa = self.tanh(aa)
            aa = torch.abs(aa)
            aa_big = aa.expand(x.size())
            xt = aa_big * x + x
        xt = self.last_layer(xt)

        sic = self.sic(xt)
        sod = self.sod(xt)
        
        df3 = self.conv13(xt)
        offset3=self.offsets3(df3)
        x_df3=self.DF3(df3,offset3)
        floe = self.floe(x_df3)

        water = self.water_decoder(encoder_list[1])
        water = self.water_head(water)

        out = {
            'SIC': sic,
            'SOD': sod,
            'FLOE': floe,
            'water':water
            }
        return out
        

if __name__ == '__main__':
    x = torch.randn(2, 14, 256, 256)
    model = MFDA_Base(18)
    aa = model(x)
    print(aa['SIC'].shape)
