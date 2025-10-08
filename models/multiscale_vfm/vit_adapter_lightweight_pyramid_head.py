#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_vfm.basic_pyramid_head import _Upscale
from models.utils.block_masking import BlockMasking
from models.vit_adapter.dinov2_vit_adapter import DinoV2ViTAdapter


class VITAdapterLightweightPyramidHead(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int = None,
            freeze_vit: bool = False,
            use_pretrained_adapter: bool = True,
            ckpt_path: str = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 14
        self.num_classes = num_classes

        assert img_size == 1024
        if img_size == 1024:
            self.img_size = 1120

        self.encoder = DinoV2ViTAdapter(
            self.img_size, model_name,
            initialize_resized_pos_embed=False,
            use_16_patch_size=False,
            return_last_vit_feature=False,
        )
        if use_pretrained_adapter and ckpt_path is None:
            urls = {
                "dinov2_vits14": "https://huggingface.co/tue-mps/vfmuda_plusplus_small_in1k_pretraining/resolve/main/dinov2_small_with_vitadapter_pretraining_in1k_step50000.ckpt",
                "dinov2_vitb14": "https://huggingface.co/tue-mps/vfmuda_plusplus_base_in1k_pretraining/resolve/main/dinov2_base_with_vitadapter_pretraining_in1k_step50000.ckpt",
                "dinov2_vitl14": "https://huggingface.co/tue-mps/vfmuda_plusplus_large_in1k_pretraining/resolve/main/dinov2_large_with_vitadapter_pretraining_in1k_step50000.ckpt",
            }
            encoder_chkpt_state_dict = torch.hub.load_state_dict_from_url(urls[model_name])['state_dict']
            encoder_chkpt_state_dict = {k.replace("network.encoder.", ""): v for k, v in encoder_chkpt_state_dict.items()}
            encoder_chkpt_state_dict = {k: v for k, v in encoder_chkpt_state_dict.items() if "head" not in k}
            self.encoder.load_state_dict(encoder_chkpt_state_dict, strict=True)

        self.upscale = _Upscale(self.encoder.embed_dim)
        out = nn.Conv2d(self.encoder.embed_dim // 8, num_classes, kernel_size=1, padding=0, bias=False)
        torch.nn.init.normal_(out.weight, 0, std=0.1)
        self.out = out


        if ckpt_path is not None:
            if "https://huggingface.co" in ckpt_path:
                chkpt_state_dict = torch.hub.load_state_dict_from_url(ckpt_path)['state_dict']
            else:
                assert os.path.exists(ckpt_path), "{}".format(ckpt_path)
                chkpt_state_dict = torch.load(ckpt_path)["state_dict"]

            chkpt_state_dict = {k: v for k, v in chkpt_state_dict.items() if ("network_feature_extractor" not in k)}
            encoder_chkpt_state_dict = {k.replace("network.encoder.", ""): v for k, v in chkpt_state_dict.items() if
                                        "network.encoder" in k}
            upscale_chkpt_state_dict = {k.replace("network.upscale.", ""): v for k, v in chkpt_state_dict.items() if
                                        "network.upscale" in k}
            out_chkpt_state_dict = {k.replace("network.out.", ""): v for k, v in chkpt_state_dict.items() if
                                    "network.out." in k}

            # do same basic key check, just to be sure
            match_factor = len(set(encoder_chkpt_state_dict).intersection(self.encoder.state_dict())) / max(
                len(set(encoder_chkpt_state_dict)), len(set(self.encoder.state_dict())))
            print("{}x of keys match when loading vit adapter checkpoint".format(match_factor))
            match_factor = len(set(upscale_chkpt_state_dict).intersection(self.upscale.state_dict())) / max(
                len(set(upscale_chkpt_state_dict)), len(set(self.upscale.state_dict())))
            print("{}x of keys match when loading upscale head checkpoint".format(match_factor))
            match_factor = len(set(out_chkpt_state_dict).intersection(self.out.state_dict())) / max(
                len(set(out_chkpt_state_dict)), len(set(self.out.state_dict())))
            print("{}x of keys match when loading out head checkpoint".format(match_factor))

            self.encoder.load_state_dict(encoder_chkpt_state_dict, strict=True)
            self.upscale.load_state_dict(upscale_chkpt_state_dict, strict=True)
            self.out.load_state_dict(out_chkpt_state_dict, strict=True)

        if freeze_vit:
            self.encoder.freeze_vit()

        self.block_masking = BlockMasking(mask_block_size=64)

        self.feat_dim = self.encoder.embed_dim

        self.param_defs_decoder = [
            ("out", self.out),
            ("upscale", self.upscale),
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

    def forward(self, img, mask_ratio=0.0, return_features=False):
        b, c, h, w = img.shape
        assert h == w
        orig_img_size = [h, w]

        img = F.interpolate(img, self.img_size, mode="bilinear", align_corners=False)

        if 0.0 < mask_ratio:
            img = self.block_masking(img, mask_ratio)

        feat_list = self.encoder(img)
        [f1, f2, f3, f4] = feat_list

        upscaled = self.upscale(feat_list)
        logit = self.out(upscaled)
        logit = F.interpolate(logit, orig_img_size, mode="bilinear", align_corners=False)
        if return_features:
            return logit, f4
        else:
            return logit

    @torch.no_grad()
    def inference(self, img):
        return self.forward(img)

    def token_to_image(self, x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x
