import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        super(Conv3dBlock, self).__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Deconv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=1, dropout_rate=0):
        super(Deconv3dBlock, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_img_shape, gf, depth):
        super(Generator, self).__init__()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        input_channels = input_img_shape[0]

        # Downsampling
        for i in range(depth):
            self.downsampling_layers.append(Conv3dBlock(input_channels, gf*(2**i)))
            input_channels = gf*(2**i)

        # Upsampling
        k = depth - 2
        for i in range(depth-1):
            self.upsampling_layers.append(Deconv3dBlock(input_channels, gf*2**k))
            input_channels = gf*2**k
            k = k - 1
        self.u4 = nn.Upsample(scale_factor=2)
        self.final_layer = nn.Conv3d(input_channels, input_img_shape[0], kernel_size=4, stride=1, padding=5)

    def forward(self, x):
        skip_connections = []
        # Downsampling
        for layer in self.downsampling_layers:
            x = layer(x)
            # skip_connections.append(x)
        # Upsampling
        for layer in self.upsampling_layers:
            x = layer(x)
            # x = torch.cat([x, skip_connection], 1)
        # Final layer
        x = self.u4(x)
        x = self.final_layer(x)
        x = x[:,:,1:,1:,1:]
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, input_img_shape, df, depth):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv3d(input_img_shape[0], df, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        input_channels = df
        for i in range(1, depth):
            layers.append(nn.Conv3d(input_channels, df*2**i, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            input_channels = df*2**i
        layers.append(nn.Conv3d(input_channels, 1, kernel_size=4, stride=2, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
