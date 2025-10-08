#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from timm.layers import resample_patch_embed
from torch.nn.init import normal_

from models.vit_adapter.adapter_modules import InteractionBlockWithCls, SpatialPriorModule, \
    deform_inputs
from models.vit_adapter.ms_deform_attn import MSDeformAttn


class DinoV2ViTAdapter(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            # pretrain_size=224,
            # num_heads=12,
            conv_inplane=64,
            n_points=4,
            # deform_num_heads=6,
            init_values=0.0,
            interaction_indexes=None,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            in_img_scale: float = 1.0,
            align_corners: bool = False,
            initialize_resized_pos_embed: bool = True,
            use_16_patch_size: bool = False,
            return_last_vit_feature: bool = False
    ):
        super().__init__()
        self.align_corners = align_corners
        self.initialize_resized_pos_embed = initialize_resized_pos_embed
        self.add_vit_feature = add_vit_feature
        self.return_last_vit_feature = return_last_vit_feature
        self.vit_in_img_size = img_size
        if use_16_patch_size:
            self.patch_size = 16
            assert (img_size % 32) == 0
        else:
            # dino needs 14 patch, vit adapter needs 16 patch
            assert (img_size % 32) == 0 and (img_size % 7) == 0
            self.patch_size = 14

        tmp_encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = tmp_encoder.embed_dim
        self.blocks = deepcopy(tmp_encoder.blocks)
        self.num_block = len(tmp_encoder.blocks)
        self.cls_token = deepcopy(tmp_encoder.cls_token)
        self.patch_embed = deepcopy(tmp_encoder.patch_embed)
        self.pos_embed = deepcopy(tmp_encoder.pos_embed)

        if use_16_patch_size:
            self.resie_patch_embed(16)

        interaction_indexes = []
        interaction_indexes_split = 4
        last_index = 0
        for i in range(
                self.num_block // interaction_indexes_split - 1,
                self.num_block,
                self.num_block // interaction_indexes_split
        ):
            interaction_indexes += [[last_index, i]]
            last_index = i + 1
        self.interaction_indexes = interaction_indexes

        if initialize_resized_pos_embed:
            self.pos_embed = torch.nn.Parameter(self.interpolate_pos_encoding(
                self.vit_in_img_size, self.vit_in_img_size))

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.interactions = nn.ModuleList(
            [
                InteractionBlockWithCls(
                    dim=self.embed_dim,
                    num_heads=self.embed_dim // 64,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                    with_cp=False,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim, momentum=0.01)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim, momentum=0.01)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim, momentum=0.01)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim, momentum=0.01)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def resie_patch_embed(self, patch_size):
        with torch.no_grad():
            new_proj = nn.Conv2d(
                self.patch_embed.proj.in_channels,
                self.patch_embed.proj.out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                bias=self.patch_embed.proj.bias is not None,
            )
            new_proj.weight.copy_(
                resample_patch_embed(self.patch_embed.proj.weight.detach().clone(), [patch_size, patch_size])
            )
            if self.patch_embed.proj.bias is not None:
                new_proj.bias.copy_(self.patch_embed.proj.bias)
            self.patch_embed.proj = new_proj
        self.patch_embed.patch_size = (patch_size, patch_size)

    def freeze_vit(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, img, mask_ratio=0.0):
        bs, c, h, w = img.shape
        assert h == w
        orig_img_size = [h, w]
        img = F.interpolate(
            img,
            size=(self.vit_in_img_size, self.vit_in_img_size),
            mode="bilinear", align_corners=self.align_corners
        )
        bs, c, h, w = img.shape

        H_c, W_c = img.shape[2] // 16, img.shape[3] // 16
        H_toks, W_toks = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size

        deform_inputs1, deform_inputs2 = deform_inputs(img, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(img)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x_patch = self.patch_embed(img)
        bs, n, dim = x_patch.shape

        if 0.0 < mask_ratio:
            raise NotImplementedError('mask ratio not implemented yet: {}'.format(mask_ratio))
        #     x_patch = self.token_masking(x_patch, mask_ratio)

        x = torch.cat((self.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        if self.initialize_resized_pos_embed:
            x = x + self.pos_embed
        else:
            x = x + self.interpolate_pos_encoding_orig(x, w, h)
        x = x.contiguous()

        # Interaction
        cls, x = (x[:, :1, :], x[:, 1:, :])
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(
                x,
                c,
                cls,
                self.blocks[indexes[0]: indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # print("c.shape", c.shape) torch.Size([2, 25725, 768])
        # Split & Reshape
        c2 = c[:, 0: c2.size(1), :]
        c3 = c[:, c2.size(1): c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            # print(c1.shape, c2.shape, c3.shape, c4.shape, x1.shape, x2.shape, x3.shape, x4.shape, H_c, H_toks)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        if self.return_last_vit_feature:
            x1, x2, x3, x4 = outs

            return [f1, f2, f3, f4], x4
        else:
            return [f1, f2, f3, f4]

    def interpolate_pos_encoding(self, w, h):
        N = self.pos_embed.shape[1] - 1
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M

        pos_embed = self.pos_embed  # .float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, self.embed_dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            antialias=False,
            align_corners=True,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        pos_embed = pos_embed.contiguous()
        return pos_embed

    def interpolate_pos_encoding_orig(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}

        # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        interpolate_offset = 0.1
        sx = float(w0 + interpolate_offset) / M
        sy = float(h0 + interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
