import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel,padding):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv3d(ch_in, ch_in, kernel_size=kernel, padding=padding, groups=ch_in)
        self.point_conv = nn.Conv3d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.downsample = nn.Sequential(
            conv1x1x1(in_planes, planes, stride),
            nn.BatchNorm3d(planes))

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU() #inplace=True
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # if(self.stride>1):
        residual = self.downsample(x)
        # residual = x
        # print("residual block:", residual.shape)
        out = self.conv1(x)
        # print("conv1:",out.shape)
        # print("out1:", out.shape)
        out = self.bn1(out)
        # print("bn:",out.shape)
        out = self.relu(out)

        out = self.conv2(out)
        # print("conv2:",out.shape)
        # print("out2:", out.shape)
        out = self.bn2(out)
        # print("bn:",out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual.clone()
        out = self.relu(out)
        # print("block out:", out.shape)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 n_input_channels,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        self.depth_conv1 = depthwise_separable_conv(n_input_channels[0], 32,3,1)
        self.depth_conv2 = depthwise_separable_conv(n_input_channels[0], 32,5,2)
        # self.depth_conv3 = depthwise_separable_conv(n_input_channels[0], 32,7,3)

        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU() #inplace=True
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU() #inplace=True
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU() #inplace=True
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv3d(n_input_channels[1],
                               n_input_channels[2],
                               # kernel_size=(conv1_t_size, 7, 7),
                               kernel_size=(3, 3, 3),
                               # stride=(conv1_t_stride, 1, 1),
                               stride=(2, 2, 2),
                               # padding=(conv1_t_size // 2, 3, 3),
                               padding=(1, 1, 1),
                               bias=False)
        self.bn4 = nn.BatchNorm3d(n_input_channels[2])
        self.relu4 = nn.ReLU() #inplace=True
        self.layer1 = BasicBlock(n_input_channels[2], n_input_channels[2], stride=2)
        self.layer2 = BasicBlock(n_input_channels[2], n_input_channels[2], stride=2)





    def forward(self, x):

        residual = x
        x1 = self.depth_conv1(x)
        x2 = self.depth_conv2(x)
        x4 = torch.cat([x1,x2], dim=1)
        x4 += residual
        x4 = self.conv1(x4)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x4 = self.layer1(x4)
        x4 = self.layer2(x4)
        return x4
