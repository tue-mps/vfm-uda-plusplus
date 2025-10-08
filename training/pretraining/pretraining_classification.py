#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import lightning
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import info
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import v2 as TV
from typing import Optional, Tuple

from datasets.utils.mappings import get_label2name
from datasets.utils.util import normalize
from models.utils.warmup_and_linear_scheduler import WarmupAndLinearScheduler
from training.utils.utils import get_full_names, get_param_group, process_parameters


class PretrainingClassification(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            img_size: int,
            network: nn.Module,
            num_classes: int,
            lr: float = 5e-5,
            lr_multiplier: float = 0.01,
            layerwise_lr_decay: float = 1.0,
            poly_lr_decay_power: float = 0.9,
            warmup_iters: int = 1500,
            weight_decay: float = 0.01,
            ignore_index: int = 255,
            lr_mode: str = "warmuplinear",
            use_strong_aug_source: bool = False,
            use_pointrend: bool = True,
            ckpt_path: Optional[str] = None,

    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.layerwise_lr_decay = layerwise_lr_decay
        self.poly_lr_decay_power = poly_lr_decay_power
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.warmup_iters = warmup_iters
        self.lr_mode = lr_mode
        self.use_strong_aug_source = use_strong_aug_source
        self.use_pointrend = use_pointrend

        self.brightness_delta = 32 / 255.
        self.contrast_delta = 0.5
        self.saturation_delta = 0.5
        self.hue_delta = 18 / 360.
        self.color_jitter_probability = 0.8

        self.save_hyperparameters()

        blur_kernel_size = int(
            np.floor(
                np.ceil(0.1 * self.img_size) - 0.5 +
                np.ceil(0.1 * self.img_size) % 2
            )
        )
        #
        random_aug_weak = TV.Compose([
            TV.RandomApply([
                TV.ColorJitter(
                    brightness=self.brightness_delta,
                    contrast=self.contrast_delta,
                    saturation=self.saturation_delta,
                    hue=self.hue_delta)], p=0.9),
        ])
        self.random_aug_weak = TV.Lambda(lambda x: torch.stack([random_aug_weak(x_) for x_ in x]))
        #
        random_aug_strong = TV.Compose([
            TV.RandomApply([TV.ColorJitter(
                brightness=0.8, contrast=0.8,
                saturation=0.8, hue=0.4)], p=0.8),
            TV.RandomGrayscale(p=0.1),
            TV.RandomApply([
                TV.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.15, 3.0))], p=0.5),
        ])
        self.random_aug_strong = TV.Lambda(lambda x: torch.stack([random_aug_strong(x_) for x_ in x]))

        self.network = network

        self.label2name = get_label2name()
        self.val_ds_names = ["val"]
        self.metrics = nn.ModuleList(
            [
                MulticlassAccuracy(num_classes=self.num_classes)
                for _ in range(len(self.val_ds_names))
            ]
        )

        self._load_ckpt(ckpt_path)
        self.automatic_optimization = False

    @torch.no_grad()
    def train_dataprep(self, batch):
        sourceds_image, sourceds_target = batch
        batch_size, _, H, W = sourceds_image.shape
        if self.use_strong_aug_source:
            sourceds_color_aug_image = self.random_aug_strong(sourceds_image)
        else:
            sourceds_color_aug_image = self.random_aug_weak(sourceds_image)

        return sourceds_image, sourceds_color_aug_image, sourceds_target

    def get_optimizers(self):
        opt = self.optimizers()
        opt.zero_grad()
        return opt

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        opt = self.get_optimizers()

        sourceds_image, sourceds_color_aug_image, sourceds_target = self.train_dataprep(batch)
        source_logits = self.network(normalize(sourceds_color_aug_image))
        loss_source = F.cross_entropy(source_logits, sourceds_target)
        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()

        self.log("loss", loss_source, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                sourceds_predicted_segmentation = torch.argmax(source_logits.detach(), dim=1)
                acc_source = (sourceds_predicted_segmentation == sourceds_target)
                acc_source = acc_source.float().mean()
                self.log("acc", acc_source, prog_bar=False)

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_target = batch
        with torch.no_grad():
            b_pred = self.network(normalize(b_image))
            b_pred = b_pred.argmax(dim=1)
        self.metrics[dataloader_idx].update(
            b_pred, b_target
        )

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def _on_eval_epoch_end(self, log_prefix):
        for metric_idx, metric in enumerate(self.metrics):
            acc = metric.compute()
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]

            self.log(
                f"{log_prefix}_{ds_name}_acc", acc, sync_dist=True
            )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):
        current_params = {
            name
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }

        lr = (
                self.lr
                * math.sqrt(self.batch_size * self.trainer.num_devices * self.trainer.num_nodes)
        )

        param_defs, current_params = process_parameters(
            self.network.param_defs_decoder, current_params
        )
        param_groups = [get_param_group(param_defs, lr)]

        lr *= self.lr_multiplier

        n_blocks = max(
            len(blocks) for _, blocks in self.network.param_defs_encoder_blocks
        )

        for i in range(n_blocks - 1, -1, -1):
            for block_name_prefix, blocks in self.network.param_defs_encoder_blocks:
                if i < len(blocks):
                    block_params = blocks[i].parameters()
                    block_param_names = get_full_names(
                        blocks[i], f"{block_name_prefix}.{i}"
                    )
                    current_params -= block_param_names
                    param_groups.append(get_param_group(block_params, lr))

            lr *= self.layerwise_lr_decay

        param_defs, current_params = process_parameters(
            self.network.param_defs_encoder_stems, current_params
        )
        param_groups.append(get_param_group(param_defs, lr))

        if current_params:
            raise ValueError(
                f"The following parameters are not included in the optimizer: {current_params}"
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        if self.lr_mode == "poly":
            lr_scheduler = {
                "scheduler": PolynomialLR(
                    optimizer,
                    int(self.trainer.estimated_stepping_batches),
                    self.poly_lr_decay_power,
                ),
                "interval": "step",
            }
        elif self.lr_mode == "warmuplinear":
            lr_scheduler = {
                "scheduler": WarmupAndLinearScheduler(
                    optimizer,
                    start_warmup_lr=1e-5,
                    warmup_iters=self.warmup_iters,
                    base_lr=1,
                    final_lr=1e-3,
                    total_iters=self.trainer.max_steps,
                ),
                "interval": "step",
            }
        else:
            raise Exception("Wrong lr_more: {}".format(self.lr_mode))

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _load_ckpt(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return

        ckpt_state: dict = torch.load(ckpt_path, map_location=self.device)

        if "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        if "model" in ckpt_state:
            ckpt_state = ckpt_state["model"]

        model_state = self.state_dict()
        skipped_keys = []
        for k in ckpt_state:
            if (k in model_state) and (ckpt_state[k].shape == model_state[k].shape):
                model_state[k] = ckpt_state[k]
            else:
                skipped_keys.append(k)

        info(f"Skipped loading keys: {skipped_keys}")

        self.load_state_dict(model_state)
