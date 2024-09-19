# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import vision_transformer3d_Baseline_SPEB


class VisionTransformer(vision_transformer3d_Baseline_SPEB.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        img = x
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # 第二个分支  每个8*8*8的块单独提取
        x2 = self.patch_embed2(img)
        x2 = x2.reshape(B, 14, 16, 14, 96)
        x2 = self.x2_reshape(x2,B)
        x2 = x2.reshape(B, 7 * 8 * 7, 768)
        # print(x2.shape)

        # 第二个分支

        # 使用reshape函数将张量重新组织成所需的形状
        x2 = x2.reshape(B, 392 * 8, 768 // 8)
        cls_token2 = self.cls_token2.expand(B, -1, -1)
        x2 = torch.cat((cls_token2, x2), dim=1)
        x2 = x2 + self.pos_embed2

        # apply Transformer blocks
        for blk in self.blocks2:
            x2 = blk(x2)
        x2 = self.norm2(x2)

        x2 = x2[:, 1:, :]
        x2 = x2.reshape(B, 392, 768)
        # x2 = self.x2_to_x1(x2)

        x[:, 1:, :] = x[:, 1:, :] + self.coefficient * x2

        for blk in self.blocks3:
            x3 = blk(x)
        x3 = self.norm3(x3)

        if self.global_pool:
            x3 = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x3)
        else:
            x3 = self.norm(x3)
            outcome = x3[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model