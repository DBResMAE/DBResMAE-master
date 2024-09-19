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
import sys

sys.path.append('/home/manjianzhi/jinjin/MRI/mae-main')
from vision_transformer3d import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
from util.pos_embed2 import get_2d_sincos_pos_embed2

from PatchEmbed3D_second import PatchEmbed2
from eca_block import EfficientChannelAttention


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(112, 128, 112), patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_attention1 = EfficientChannelAttention(num_patches)  # 注意通道数
        self.patch_attention2 = EfficientChannelAttention(num_patches * 0.25)  # 注意通道数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, num_patches=99,
                  batch_size=1)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 第二分支
        # 定义一个可学习的参数系数
        self.coefficient = nn.Parameter(torch.tensor(1.0))
        self.patch_embed2 = PatchEmbed2(img_size, 8, embed_dim // 8)
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim // 8))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches * 8 + 1, embed_dim // 8),
                                       requires_grad=False)  # fixed sin-cos embedding
        self.blocks2 = nn.ModuleList([
            Block(embed_dim // 8, 6, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, num_patches=785)
            for i in range(6)])
        self.norm2 = norm_layer(embed_dim // 8)
        self.x2_to_x1 = nn.Linear(embed_dim, embed_dim, bias=True)

        self.blocks3 = nn.ModuleList([
            Block(embed_dim, 2, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, num_patches=99)
            for i in range(2)])
        self.norm3 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  num_patches=393, batch_size=1)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        pos_embed2 = get_2d_sincos_pos_embed2(self.pos_embed2.shape[-1], int(self.patch_embed2.num_patches ** .5),
                                              cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed2.data.copy_(torch.from_numpy(pos_embed2).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True, flag=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W, D)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]

        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p
        h, w, d = imgs.shape[2] // p, imgs.shape[3] // p, imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = 7
        w = 8
        d = 7
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, x2, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 设置随机数种子
        # torch.manual_seed(42)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x2_masked = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x2_masked, mask, ids_restore

    def x2_reshape(self, x2):
        # 初始化一个空的结果张量，大小为（7，8，7，96*8）
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        result_tensor = torch.zeros(1, 7, 8, 7, 96 * 8, device=device)

        # 遍历输入张量，按要求拼接
        for b in range(1):
            for i in range(7):
                for j in range(8):
                    for k in range(7):
                        # 切片提取块
                        block = x2[b, i:i + 2, j:j + 2, k:k + 2, :]

                        # 拼接到目标张量中
                        result_tensor[b, i, j, k, :] = block.reshape(-1, 96 * 8).reshape(-1)

        return result_tensor

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x1 = self.patch_embed(x)

        eca_out = self.patch_attention1(x1)
        x1 = x1 * eca_out

        # add pos embed w/o cls token
        x1 = x1 + self.pos_embed[:, 1:, :]

        # 第二个分支  每个8*8*8的块单独提取
        x2 = self.patch_embed2(x)
        x2 = x2 + self.pos_embed2[:, 1:, :]
        x2 = x2.reshape(1, 14, 16, 14, 96)
        x2 = self.x2_reshape(x2)

        x2 = x2.reshape(1, 7 * 8 * 7, 768)

        # masking: length -> length * mask_ratio
        x1, x2, mask, ids_restore = self.random_masking(x1, x2, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)

        # 第二个分支

        # 使用reshape函数将张量重新组织成所需的形状
        x2 = x2.reshape(1, 98 * 8, 768 // 8)
        cls_token2 = self.cls_token2 + self.pos_embed2[:, :1, :]
        cls_tokens2 = cls_token2.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_tokens2, x2), dim=1)

        # apply Transformer blocks
        for blk in self.blocks2:
            x2 = blk(x2)
        x2 = self.norm2(x2)

        x2 = x2[:, 1:, :]
        x2 = x2.reshape(1, 98, 768)

        # x2 = self.x2_to_x1(x2)
        # 使用 torch.clamp 函数确保参数系数在 [0, 1] 范围内
        coefficient_clamped = torch.clamp(self.coefficient, 0, 1)

        x1[:, 1:, :] = x1[:, 1:, :] + coefficient_clamped * x2

        for blk in self.blocks3:
            x3 = blk(x1)
        x3 = self.norm3(x3)

        return x3, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        eca_out = self.patch_attention1(x[:, 1:, :])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tensor_eca = torch.ones(x.shape[0], x.shape[1], x.shape[2], device=device)
        tensor_eca[:, 1:, :] = eca_out
        x = x * tensor_eca

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        # 还原图像
        imgs = self.unpatchify(x)

        return imgs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        img = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask)
        return img


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
