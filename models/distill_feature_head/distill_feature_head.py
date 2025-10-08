#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


class DistillFeatureHead(torch.nn.Module):

    def __init__(self, in_embed_dim, out_embed_dim):
        super().__init__()

        self.feat_project = nn.ModuleList([
            nn.Conv2d(in_embed_dim, in_embed_dim * 4, kernel_size=1, padding=0),
            LayerNorm2d(in_embed_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_embed_dim * 4, out_embed_dim, kernel_size=1, padding=0),
        ])

    def forward(self, x, out_size=None):
        for blk in self.feat_project:
            x = blk(x)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x

