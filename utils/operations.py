from enum import Enum
from typing import Optional

import torch.nn as nn

from utils.utils import *

OPS = {
    'none': lambda c_in, c_ot, op_type, dp: AdapterBlock(c_in, c_ot, ZeroOp(stride=1)),
    'identity': lambda c_in, c_ot, op_type, dp: AdapterBlock(c_in, c_ot, nn.Identity()),
    'avg_pool': lambda c_in, c_ot, op_type, dp: build_ops('avg_pool', op_type, c_in, c_ot),
    'max_pool': lambda c_in, c_ot, op_type, dp: build_ops('max_pool', op_type, c_in, c_ot),
    'up_sample': lambda c_in, c_ot, op_type, dp: AdapterBlock(c_in, c_ot, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),

    'conv_3': lambda c_in, c_ot, op_type, dp: build_ops('conv_3', op_type, c_in, c_ot, dp=dp),
    'se_conv_3': lambda c_in, c_ot, op_type, dp: build_ops('se_conv_3', op_type, c_in, c_ot, dp=dp),
    'dil_3_conv_5': lambda c_in, c_ot, op_type, dp: build_ops('dil_3_conv_5', op_type, c_in, c_ot, dp=dp),
    'dil_2_conv_5': lambda c_in, c_ot, op_type, dp: build_ops('dil_2_conv_5', op_type, c_in, c_ot, dp=dp),
    'dep_sep_conv_3': lambda c_in, c_ot, op_type, dp: build_ops('dep_sep_conv_3', op_type, c_in, c_ot, dp=dp),
    'dep_sep_conv_5': lambda c_in, c_ot, op_type, dp: build_ops('dep_sep_conv_5', op_type, c_in, c_ot, dp=dp),
}

DownOps = [
    'avg_pool',
    'se_conv_3',
    'dil_3_conv_5',
    'dil_2_conv_5',
    'dep_sep_conv_3',
    'dep_sep_conv_5',
]

UpOps = [
    'up_sample',
    'se_conv_3',
    'dil_3_conv_5',
    'dil_2_conv_5',
    'dep_sep_conv_3',
    'dep_sep_conv_5',
]

NormOps = [
    'identity',
    'none',
    'dil_3_conv_5',
    'dil_2_conv_5',
    'dep_sep_conv_3',
    'dep_sep_conv_5',
]


class OpType(Enum):
    UP = {'id': 1, 'ops': UpOps}
    DOWN = {'id': 2, 'ops': DownOps}
    NORM = {'id': 3, 'ops': NormOps}


def build_ops(op_name, op_type: OpType, c_in: Optional[int] = None, c_ot: Optional[int] = None, dp=0):
    stride = 1 if op_type == OpType.NORM else 2
    use_transpose = True if op_type == OpType.UP else False
    output_padding = 1 if op_type == OpType.UP else 0
    if op_name == 'avg_pool':
        return AdapterBlock(c_in, c_ot, nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False))
    elif op_name == 'max_pool':
        return AdapterBlock(c_in, c_ot, nn.MaxPool2d(3, stride=stride, padding=1))
    elif op_name == 'conv_3':
        return ConvBn(c_in, c_ot, kernel_size=3, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    elif op_name == 'se_conv_3':
        return ConvBnSe(c_in, c_ot, kernel_size=3, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    elif op_name == 'dil_3_conv_5':
        return ConvBn(c_in, c_ot, kernel_size=5, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=3, dropout=dp)
    elif op_name == 'dil_2_conv_5':
        return ConvBn(c_in, c_ot, kernel_size=5, stride=stride, transpose=use_transpose, output_padding=output_padding, dilation=2, dropout=dp)
    elif op_name == 'dep_sep_conv_3':
        return DepSepConv(c_in, c_ot, kernel_size=3, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    elif op_name == 'dep_sep_conv_5':
        return DepSepConv(c_in, c_ot, kernel_size=5, stride=stride, transpose=use_transpose, output_padding=output_padding, dropout=dp)
    else:
        raise NotImplementedError()


class ReLUConv(nn.Sequential):

    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0, dropout=0):
        act = build_activation(False)
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        super().__init__(act, *conv)


class ConvBn(nn.Sequential):

    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        norm = build_norm(c_ot, affine)
        super().__init__(*conv, norm)


class ConvBnSe(nn.Sequential):
    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        conv = build_weight(c_in, c_ot, kernel_size, stride, dilation, transpose, output_padding, dropout)
        norm = build_norm(c_ot, affine)
        se = SEBlock(c_ot)
        super().__init__(*conv, norm, se)


class DepSepConv(nn.Sequential):
    def __init__(self, c_in, c_ot, kernel_size=3, stride=1, dilation=1, transpose=False, output_padding=0,
                 affine=True, dropout=0):
        depth_conv = build_weight(c_in, c_in, kernel_size, stride, dilation, transpose, output_padding, dropout, groups=c_in)
        depth_norm = build_norm(c_in, affine)
        depth_act = build_activation()
        point_conv = build_weight(c_in, c_ot, 1, 1, 1, False, 0, dropout)
        point_norm = build_norm(c_ot, affine)
        super().__init__(*depth_conv, depth_norm, depth_act, *point_conv, point_norm)


def build_weight(c_in, c_ot, kernel_size, stride, dilation, use_transpose, output_padding, dropout_rate, groups=1):
    padding = get_same_padding(kernel_size)
    padding *= dilation
    ops = []
    if dropout_rate > 0:
        ops.append(nn.Dropout2d(dropout_rate, inplace=False))
    if use_transpose:
        ops.append(nn.ConvTranspose2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=False, output_padding=output_padding, groups=groups))
    else:
        ops.append(nn.Conv2d(c_in, c_ot, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                             bias=False, groups=groups))
    return ops


def build_norm(c_ot, affine):
    return nn.BatchNorm2d(c_ot, affine=affine)


def build_activation(inplace=True):
    return nn.ReLU(inplace=inplace)


def build_rectify(c_in, c_ot, cell_type):
    act = build_activation(False)
    if cell_type == 'up':
        if c_in == c_ot:
            return nn.Sequential(act, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), build_norm(c_ot, True))
        else:
            return nn.Sequential(act, nn.ConvTranspose2d(c_in, c_ot, kernel_size=1, stride=2, output_padding=1, bias=False), build_norm(c_ot, True))
    else:
        if c_in == c_ot:
            return nn.Sequential(act, nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False), build_norm(c_ot, True))
        else:
            return nn.Sequential(act, nn.Conv2d(c_in, c_ot, kernel_size=1, stride=2, bias=False), build_norm(c_ot, True))


class ZeroOp(nn.Module):

    def __init__(self, stride):
        super(ZeroOp, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class AdapterBlock(nn.Module):

    def __init__(self, c_in, c_ot, module):
        super().__init__()
        self.c_in = c_in
        self.c_ot = c_ot
        self.module = module
        if self.c_in != self.c_ot:
            self.conv = nn.Conv2d(c_in, c_ot, kernel_size=1, bias=False)
        self.norm = build_norm(c_ot, True)

    def forward(self, x):
        out = self.module(x)
        if self.c_in != self.c_ot:
            out = self.conv(out)
        out = self.norm(out)
        return out


class SEBlock(nn.Module):
    # credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    def __init__(self, c, r=16):
        super().__init__()
        self.mid = c // r if c > r else 1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, self.mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class ShrinkBlock(nn.Module):

    def __init__(self, c_in, c_ot):
        super().__init__()
        self.act = build_activation(False)
        self.conv = nn.Conv2d(c_in, c_ot, kernel_size=3, padding=1, bias=False)
        self.norm = build_norm(c_ot, True)

    def forward(self, x):
        out = self.act(x)
        out = self.conv(out)
        out = self.norm(out)
        return out


class RectifyBlock(nn.Module):

    def __init__(self, c_in, c_ot, cell_type='down'):
        super().__init__()
        self.cell_type = cell_type
        self.conv = nn.Conv2d(c_in, c_ot, kernel_size=3, padding=1, bias=False)
        self.norm = build_norm(c_ot, True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return out


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

