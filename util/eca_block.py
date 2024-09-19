import torch
import torch.nn as nn
import math

class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        # print(x.shape)
        out = self.sigmoid(x)
        # print(out.shape)
        # exit(0)
        return out