#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import math
import torch.nn.init
from torch import nn


class _Upscale(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.up_conv1 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0, bias=True)
        self.up_conv2 = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 4, kernel_size=2, stride=2, padding=0, bias=True)
        self.up_conv3 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, kernel_size=2, stride=2, padding=0, bias=True)

        self.reduce_conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=True)
        self.reduce_conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=True)
        self.reduce_conv3 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=True)

        self.smooth_conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, bias=True, padding=1)
        self.smooth_conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, bias=True, padding=1)
        self.smooth_conv3 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, bias=True, padding=1)
        self.smooth_conv4 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, bias=True, padding=1)

        self.norm1 = nn.SyncBatchNorm(in_channels // 2, momentum=0.01)
        self.norm2 = nn.SyncBatchNorm(in_channels // 4, momentum=0.01)
        self.norm3 = nn.SyncBatchNorm(in_channels // 8, momentum=0.01)

        self.act = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        [f1, f2, f3, f4] = x  # 4 // 16, 2 // 16, 1 // 16, 1 // 32
        up1 = self.smooth_conv1(self.up_conv1(f4) + self.reduce_conv1(f3))
        up1 = self.act(self.norm1(up1))

        up2 = self.smooth_conv2(self.up_conv2(up1) + self.reduce_conv2(f2))
        up2 = self.act(self.norm2(up2))

        up3 = self.smooth_conv3(self.up_conv3(up2) + self.reduce_conv3(f1))
        up3 = self.act(self.norm3(up3))

        up4 = self.smooth_conv4(up3)
        return up4
