import torch
import torch.nn as nn
from  TransBTS.Unet_skipconnection import Unet
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

class PatchEmbed2(nn.Module):
    def __init__(self, img_size=(224, 224, 224), patch_size=16, embedding_dim=512):
        super(PatchEmbed2, self).__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[0]) * (img_size[2] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(1, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W, D = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)


        return x

