import torch.nn as nn
from timm.models.layers import DropPath
# from torchvision.models import resnet34
import torch.nn.functional as F
# import numpy as np
import torch
import numpy as np
from math import sqrt


# the Implementation of MCA
class Multi_CrossAttention(nn.Module):

    def __init__(self, hidden_size, all_head_size, head_num=4, mask=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.attention_mask = mask

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y):  # ,attention_mask):

        x_b, x_c = x.shape
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(
            batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(
            batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(
            batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # self.attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s, k_s, v_s, self.attention_mask)
        # attention: [batch_size, seq_length, num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.h_size)

        # output: [batch_size, seq_length, hidden_size]
        output = self.linear_output(attention)
        output = output.reshape(x_b, x_c)

        return output


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))

        # attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


# This code is learned from yolov9  https://github.com/WongKinYiu/yolov9.git
class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):  # CBS实现
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim),
                                       groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros(
                    (self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(
                    kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, kernels, groups, expand
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

##### CBNet #####


class CBLinear(nn.Module):
    # ch_in, ch_outs, kernel, stride, padding, groups
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s,
                              autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest')
               for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

#################


class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
# -----------------------------------------------------------------------------------------


# this code is learned from EMTCAL https://github.com/TangXu-Group/Remote-Sensing-Images-Classification
class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
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
    "Implementation of self-attention"

    def __init__(self, dim=512, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr1 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=1, stride=2)
        self.sr2 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3,
                             stride=sr_ratio, padding=1, dilation=1)
        self.sr3 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3,
                             stride=sr_ratio, padding=2, dilation=2)
        self.sr4 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3,
                             stride=sr_ratio, padding=3, dilation=3)

        self.pos1 = PosCNN(int(dim/4), int(dim/4))
        self.pos2 = PosCNN(int(dim/4), int(dim/4))
        self.pos3 = PosCNN(int(dim/4), int(dim/4))
        self.pos4 = PosCNN(int(dim/4), int(dim/4))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** (1/2))
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)
        x = x.transpose(-2, -1).reshape(B, C, H, W)
        x1 = x[:, :int(C/4), :, :]
        x2 = x[:, int(C/4):int(C/4)*2, :, :]
        x3 = x[:, int(C/4)*2:int(C/4)*3, :, :]
        x4 = x[:, int(C/4)*3:int(C/4)*4, :, :]

        sr1 = self.sr1(x1).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr2 = self.sr2(x2).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr3 = self.sr3(x3).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr4 = self.sr4(x4).reshape(B, int(C/4), -1).permute(0, 2, 1)
#         print(sr4.shape)
        sr_h = sr_w = int(sr4.shape[1] ** (1/2))
        sr1 = self.pos1(sr1, sr_h, sr_w)
        sr2 = self.pos2(sr2, sr_h, sr_w)
        sr3 = self.pos3(sr3, sr_h, sr_w)
        sr4 = self.pos4(sr4, sr_h, sr_w)

        x_sr = torch.cat((sr1, sr2, sr3, sr4), dim=2)
        x_sr = self.norm(x_sr)
        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
#         print(q.shape)
#         print(k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 3, s,
                              1, bias=True, groups=embed_dim)
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Implementation of Transformer,
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, f_kv, f3):
        x0 = f3
        x_b, x_c, x_h, x_w = x.shape
        x = x.flatten(2).transpose(2, 1)  # 经过交叉注意力机制后展平(1,49,512)
        f_kv = f_kv.flatten(2).transpose(2, 1)
        B, N, C = x.shape
        # print(x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(f_kv).reshape(B, N, 2, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
#         print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2, 1).reshape(x_b, x_c, x_h, x_w)
        out = out + x0

        return out  # shape(1,512,7,7)


def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x
