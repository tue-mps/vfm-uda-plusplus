#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoV2FeatureExtractor(nn.Module):
    def __init__(
            self,
            model_name: str,
    ):
        super().__init__()
        self.patch_size = 14
        # we fix img size so dinov2 runs at optimal resolution
        self.vit_in_img_size = 406
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward_features(self, img: torch.Tensor):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)
        x = torch.cat((self.encoder.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        x = x + self.encoder.interpolate_pos_encoding(x, w, h)

        if self.encoder.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.encoder.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        x = x.contiguous()
        for i in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = self.token_to_image(x, token_img_shape)
        return x

    @torch.no_grad()
    def forward(self, img):
        b, c, h, w = img.shape
        assert h == w

        img = F.interpolate(
            img,
            size=(self.vit_in_img_size, self.vit_in_img_size),
            mode="bilinear", align_corners=False
        )
        feats = self.forward_features(img)
        return feats

    def token_to_image(self, x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1 + self.encoder.num_register_tokens:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x
