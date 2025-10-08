#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_adapter.dinov2_vit_adapter import DinoV2ViTAdapter


class VITAdapterClassification(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int,
            patch_size: int = 14,
            freeze_vit=True,
            align_corners=False,
            use_16_patch_size=False
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.align_corners = align_corners
        assert img_size == 1024

        if use_16_patch_size:
            self.in_img_size = 1024
        else:
            self.in_img_size = 1120

        self.encoder = DinoV2ViTAdapter(
            self.in_img_size, model_name,
            initialize_resized_pos_embed=False,
            use_16_patch_size=use_16_patch_size
        )

        if freeze_vit:
            self.encoder.freeze_vit()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, 3),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.SyncBatchNorm(self.encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.SyncBatchNorm(self.encoder.embed_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.embed_dim, self.num_classes),
        )

        self.param_defs_decoder = [
            ("head", self.head),
            ("encoder.level_embed", self.encoder.level_embed),
            ("encoder.spm", self.encoder.spm),
            ("encoder.interactions", self.encoder.interactions),
            ("encoder.up", self.encoder.up),
            ("encoder.norm1", self.encoder.norm1),
            ("encoder.norm2", self.encoder.norm2),
            ("encoder.norm3", self.encoder.norm3),
            ("encoder.norm4", self.encoder.norm4),
        ]

        self.param_defs_encoder_blocks = [
            ("encoder.blocks", self.encoder.blocks),
        ]

        self.param_defs_encoder_stems = [
            # ("encoder.mask_token", self.encoder.mask_token),
            # ("encoder.norm", self.encoder.norm),
            ("encoder.pos_embed", self.encoder.pos_embed),
            ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
            ("encoder.cls_token", self.encoder.cls_token)
            if hasattr(self.encoder, "cls_token")
            else (None, None),
        ]

    def forward(self, img):
        b, c, h, w = img.shape
        assert h == w
        orig_img_size = [h, w]
        img = F.interpolate(img, self.in_img_size, mode="bilinear", align_corners=self.align_corners)
        feat_list = self.encoder(img)
        [f1, f2, f3, f4] = feat_list
        logit = self.head(f4)  # [:, :, 0, 0]
        return logit

    @torch.no_grad()
    def inference(self, img):
        return self.forward(img)
