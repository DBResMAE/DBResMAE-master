import torch
import torch.nn as nn
from  TransBTS.Unet_skipconnection import Unet
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from resnet import ResNet, BasicBlock


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224, 224), patch_size=16, embedding_dim=512):
        super(PatchEmbed, self).__init__()
        self.embedding_dim = embedding_dim
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[0]) * (img_size[2] // patch_size[0])
        self.num_patches = num_patches
        self.patch_size = patch_size

        # self.Unet = Unet(in_channels=1, base_channels=8)  #原始resnet做embed需要打开注释
        # 初始化resnetgan网络，用于特征提取
        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], [16, 32, 64, 128])
        # self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], [96, 192, 64, 128])
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv_x = nn.Conv3d(128, self.embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        # print(x.shape)
        # exit(0)
        x1 = x.permute(0, 2, 3, 4, 1).contiguous()
        x2 = x1.view(x1.size(0), -1, self.embedding_dim)

        # return x2, x1
        return x2

