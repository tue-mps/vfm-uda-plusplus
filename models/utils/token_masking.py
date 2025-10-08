#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#


import torch
from torch import nn


class TokenMasking(nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = mask_token

    def forward(self, x, mask_ratio):
        if self.training:
            keep_mask = self.get_random_token_mask_idx(x, mask_ratio)
            x = torch.where(keep_mask.unsqueeze(-1), x, self.mask_token.to(x.dtype).unsqueeze(0))
            x = x.contiguous()
            return x
        else:
            return x

    def get_random_token_mask_idx(self, x: torch.Tensor, mask_ratio: float):
        B, L, C = x.shape
        noise = torch.rand((B, L), device=x.device)
        keep_mask = mask_ratio < noise
        return keep_mask
