#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import copy
import io
import lightning
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from logging import info
from timm.layers import DropPath, PatchDropout
from torch.nn.modules.dropout import _DropoutNd
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import transforms as TV
from torchvision.transforms import functional as FV
from typing import Optional, Tuple

from datasets.utils.mappings import get_label2name
from datasets.utils.util import colorize_mask, normalize, normalize_inverse
from models.distill_feature_head.dinov2_feature_extractor import DinoV2FeatureExtractor
from models.distill_feature_head.distill_feature_head import DistillFeatureHead
from models.utils.warmup_and_linear_scheduler import WarmupAndLinearScheduler
from training.utils.calculate_uncertainty import calculate_uncertainty
from training.utils.dacs_utils import get_class_masks, one_mix
from training.utils.inference_collection import slide_inference_pre_cropped
from training.utils.utils import get_full_names, get_param_group, process_parameters


class UDA(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            img_size: int,
            network: nn.Module,
            lr: float,
            lr_multiplier: float,
            layerwise_lr_decay,
            warmup_iters: int,
            weight_decay: float,
            token_mask_ratio: float,
            fd_model_name: str,
            use_strong_aug_source: bool,
            ema_alpha: float,
            fd_weight_source: float,
            fd_weight_target: float,
            ignore_index: int = 255,
            final_lr: float = 0.0,
    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.final_lr = final_lr
        self.lr_multiplier = lr_multiplier
        self.layerwise_lr_decay = layerwise_lr_decay
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.num_classes = network.num_classes
        self.warmup_iters = warmup_iters
        self.use_strong_aug_source = use_strong_aug_source
        self.brightness_delta = 32 / 255.
        self.contrast_delta = 0.5
        self.saturation_delta = 0.5
        self.hue_delta = 18 / 360.

        ##
        # uda specific params
        ##
        self.ema_alpha = ema_alpha
        self.color_jitter_strength = 0.2
        self.color_jitter_probability = 0.8
        self.pseudo_threshold = 0.968
        self.token_mask_ratio = token_mask_ratio

        ##
        # feature distance params
        ##
        self.fd_weight_source = fd_weight_source
        self.fd_weight_target = fd_weight_target
        self.fd_model_name = fd_model_name

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


        #
        random_aug_uda = TV.Compose([
            TV.RandomApply([
                TV.ColorJitter(
                    brightness=self.color_jitter_strength,
                    contrast=self.color_jitter_strength,
                    saturation=self.color_jitter_strength,
                    hue=self.color_jitter_strength)], p=self.color_jitter_probability),
            TV.RandomApply([
                TV.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.15, 1.15))], p=0.5),
        ])
        self.random_aug_uda = TV.Lambda(lambda x: torch.stack([random_aug_uda(x_) for x_ in x]))

        self.network = network
        self.network_ema = copy.deepcopy(network)
        for p in self.network_ema.parameters():
            p.requires_grad = False

        if 0 < fd_weight_source and 0 < fd_weight_target:
            self.network_feature_extractor = DinoV2FeatureExtractor(fd_model_name)
            for p in self.network_feature_extractor.parameters():
                p.requires_grad = False
            self.network_feature_head = DistillFeatureHead(
                self.network.feat_dim,
                self.network_feature_extractor.encoder.embed_dim,
            )

        self.label2name = get_label2name()
        self.val_ds_names = ["cityscapes", "bdd", "mapillary", "wilddash", "acdc", "darkzurich"]
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_index,
                    average=None,
                )
                for _ in range(len(self.val_ds_names))
            ]
        )

        self.automatic_optimization = False

    @torch.no_grad()
    def train_dataprep(self, batch):
        sourceds_image, sourceds_target, sourceds_ignore_mask, targetds_image, targetds_target, targetds_ignore_mask = batch
        batch_size, _, H, W = sourceds_image.shape

        sourceds_image = sourceds_image.float() / 255
        targetds_image = targetds_image.float() / 255

        if torch.rand(1) < 0.5:
            sourceds_image, sourceds_target = FV.hflip(sourceds_image), FV.hflip(sourceds_target)
        if torch.rand(1) < 0.5:
            targetds_image, targetds_target = FV.hflip(targetds_image), FV.hflip(targetds_target)

        if self.use_strong_aug_source:
            sourceds_aug_image = self.random_aug_strong(sourceds_image)
        else:
            sourceds_aug_image = self.random_aug_weak(sourceds_image)

        pseudo_label, pseudo_weight, uncertainty = self.get_pseudo_label_and_weight(
            targetds_image, targetds_ignore_mask)
        mixed_img, mixed_lbl, mixed_seg_weight = self.get_mixed_source_and_target(
            sourceds_image, sourceds_target,
            targetds_image,
            pseudo_label, pseudo_weight, uncertainty
        )

        return (
            (sourceds_image, sourceds_aug_image, sourceds_target, sourceds_ignore_mask),
            (targetds_image, targetds_target, targetds_ignore_mask),
            (mixed_img, mixed_lbl, mixed_seg_weight),
            (pseudo_label, pseudo_weight, uncertainty),
        )

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

        ##
        # data prep
        ##
        sourceds_stuff, targetds_stuff, mixed_stuff, other_stuff = self.train_dataprep(batch)

        sourceds_image, sourceds_aug_image, sourceds_target, sourceds_ignore_mask = sourceds_stuff
        targetds_image, targetds_target, targetds_ignore_mask = targetds_stuff
        mixed_img, mixed_lbl, mixed_seg_weight = mixed_stuff
        pseudo_label, pseudo_weight, uncertainty = other_stuff

        ##
        # supervised training
        ##
        source_logits = self.network(normalize(sourceds_aug_image))
        loss_source = F.cross_entropy(source_logits, sourceds_target, ignore_index=self.ignore_index)
        self.manual_backward(loss_source)
        loss_source = loss_source.detach()
        source_logits = source_logits.detach()

        ##
        # uda training
        ##
        mixed_logits = self.network(normalize(mixed_img))

        assert mixed_lbl.shape[1] == 1
        loss_mix = F.cross_entropy(mixed_logits, mixed_lbl[:, 0, :, :], ignore_index=self.ignore_index,
                                   reduction='none')
        loss_mix = (loss_mix * mixed_seg_weight).mean()

        self.manual_backward(loss_mix)
        loss_mix = loss_mix.detach()

        if 0.0 < self.token_mask_ratio:
            targetds_color_aug_image = self.random_aug_uda(targetds_image)
            masked_logits = self.network(normalize(targetds_color_aug_image), mask_ratio=self.token_mask_ratio)
            loss_masked = F.cross_entropy(masked_logits, pseudo_label, ignore_index=self.ignore_index,
                                          reduction='none')
            loss_masked = (loss_masked * pseudo_weight).mean()
            self.manual_backward(loss_masked)
            loss_masked = loss_masked.detach()
        else:
            loss_masked = 0

        ##
        # fdist
        ##
        loss_fd_source = 0.
        loss_fd_target = 0.
        if 0 < self.fd_weight_source:
            loss_fd_source = self.get_feature_distance_loss(sourceds_image) * self.fd_weight_source
            self.manual_backward(loss_fd_source)
            loss_fd_source = loss_fd_source.detach()
        if 0 < self.fd_weight_target:
            loss_fd_target = self.get_feature_distance_loss(targetds_image) * self.fd_weight_target
            self.manual_backward(loss_fd_target)
            loss_fd_target = loss_fd_target.detach()
        loss_fd = loss_fd_source + loss_fd_target

        opt.step()
        self.lr_schedulers().step()

        ##
        # logging
        ##
        loss_total = loss_source + loss_mix + loss_masked + loss_fd

        self.log("train_loss_source", loss_source, prog_bar=False)
        self.log("train_loss_mix", loss_mix, prog_bar=False)
        self.log("loss_masked", loss_masked, prog_bar=False)
        self.log("train_loss_fdist", loss_fd, prog_bar=False)
        self.log("train_loss_total", loss_total, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 5) == 0:
                targetds_pseudo_lbl = pseudo_label

                accept_mask = sourceds_target != self.ignore_index
                sourceds_predicted_segmentation = torch.argmax(source_logits, dim=1)
                acc_source = (sourceds_predicted_segmentation == sourceds_target)
                acc_source = (acc_source * accept_mask).sum() / accept_mask.sum()
                self.log("train_acc_source", acc_source, prog_bar=True)

                accept_mask = targetds_target != self.ignore_index
                acc_target = (targetds_pseudo_lbl == targetds_target)
                acc_target = (acc_target * accept_mask).sum() / accept_mask.sum()
                self.log("train_acc_target", acc_target, prog_bar=True)

            if (self.global_step % 100) == 0:
                mixed_lbl = mixed_lbl[:, 0, :, :]

                self._log_train(
                    sourceds_aug_image, sourceds_target,
                    targetds_image, targetds_target, targetds_ignore_mask,
                    mixed_img, mixed_lbl, mixed_seg_weight,
                    sourceds_predicted_segmentation,
                    targetds_pseudo_lbl,
                    torch.argmax(mixed_logits.detach(), dim=1),
                    uncertainty.detach()
                )

        with torch.no_grad():
            alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
            for network_ema, network in zip(self.network_ema.parameters(), self.network.parameters()):
                network_ema.data.mul_(alpha).add_((1 - alpha) * network.detach().data)

    def get_feature_distance_loss(self, image):
        with torch.no_grad():
            self.network_feature_extractor.eval()
            for m in self.network_feature_extractor.modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
                if type(m).__name__ == "DropPath":
                    m.training = False
                if isinstance(m, PatchDropout):
                    m.training = False

            feats_frozen = self.network_feature_extractor(normalize(image))
            h, w = feats_frozen.shape[-2:]

        _, feats = self.network(normalize(self.random_aug_weak(image)), return_features=True)
        feats = self.network_feature_head(feats, (h, w))

        diff = F.smooth_l1_loss(feats, feats_frozen, reduction="none").mean(1) * 0.1
        diff += (1 - F.cosine_similarity(feats, feats_frozen)) * 0.9

        loss_featup = diff.mean()

        return loss_featup

    @torch.no_grad()
    def get_pseudo_label_and_weight(self, targetds_image, targetds_ignore_mask):
        targetds_image = normalize(targetds_image.detach().clone())
        self.network_ema.train()
        for m in self.network_ema.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
            if type(m).__name__ == "DropPath":
                m.training = False
            if isinstance(m, PatchDropout):
                m.training = False

        ema_logit_1 = self.network_ema(targetds_image)
        ema_logit_2 = FV.hflip(self.network_ema(FV.hflip(targetds_image)))
        ema_logit = (ema_logit_1 + ema_logit_2) / 2

        ema_softmax_1 = torch.softmax(ema_logit_1, dim=1)
        ema_softmax_2 = torch.softmax(ema_logit_2, dim=1)
        ema_softmax = (ema_softmax_1 + ema_softmax_2) / 2

        pseudo_prob, _ = torch.max(ema_softmax, dim=1)

        uncertainty = -calculate_uncertainty(ema_logit).squeeze(1)
        uncertainty = uncertainty / (uncertainty.amax(dim=(1, 2), keepdim=True) + 1e-12)

        # get pseudo label
        pseudo_label = torch.argmax(ema_softmax, dim=1)

        # get weight for pseudo label
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        pseudo_weight = torch.mean(ps_large_p.float()).item()

        pseudo_weight = pseudo_weight * torch.ones_like(pseudo_prob)
        pseudo_weight[targetds_ignore_mask == 255] = 0

        pseudo_weight, uncertainty = pseudo_weight.to(ema_softmax), uncertainty.to(ema_softmax)

        return pseudo_label.contiguous(), pseudo_weight.contiguous(), uncertainty.detach().clone().contiguous()

    @torch.no_grad()
    def get_mixed_source_and_target(
            self,
            sourceds_image, sourceds_target,
            targetds_image,
            pseudo_label, pseudo_weight, uncertainty
    ):
        batch_size = sourceds_image.size(0)
        gt_pixel_weight = (sourceds_target != 255).to(pseudo_weight)
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(sourceds_target)
        mixed_weight = torch.zeros_like(pseudo_weight)

        targetds_image = self.random_aug_weak(targetds_image)
        for i in range(batch_size):
            tmp_mixed_img, mixed_lbl[i] = one_mix(
                mask=mix_masks[i],
                data=torch.stack((sourceds_image[i], targetds_image[i])),
                target=torch.stack((sourceds_target[i], pseudo_label[i])))
            _, mixed_weight[i] = one_mix(
                mask=mix_masks[i],
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

            # apply diff aug for each batch image
            mixed_img[i] = self.random_aug_uda(tmp_mixed_img)

        # no grad, uniform memory
        mixed_img = torch.cat(mixed_img).detach().clone().contiguous()
        mixed_lbl = torch.cat(mixed_lbl).detach().clone().contiguous()
        mixed_weight = mixed_weight.detach().clone().contiguous()

        return mixed_img, mixed_lbl, mixed_weight

    @torch.no_grad()
    def _log_train(
            self,
            sourceds_image, sourceds_target,
            targetds_image, targetds_target, targetds_ignore_mask,
            mixed_img, mixed_lbl, mixed_seg_weight,
            predicted_source_student,
            predicted_target_ema,
            predicted_mix_student,
            uncertainty
    ):
        sourceds_image = (sourceds_image[0]).cpu().permute(1, 2, 0).float().numpy()
        color_sourceds_target = colorize_mask(sourceds_target[0].cpu().long().numpy())

        targetds_image = (targetds_image[0]).cpu().permute(1, 2, 0).float().numpy()
        color_targetds_target = colorize_mask(targetds_target[0].cpu().long().numpy())
        targetds_ignore_mask = targetds_ignore_mask[0].cpu().float().numpy()

        mixed_img = (mixed_img[0]).cpu().permute(1, 2, 0).float().numpy()
        color_mixed_lbl = colorize_mask(mixed_lbl[0].cpu().long().numpy())
        mixed_seg_weight = mixed_seg_weight[0].cpu().float().numpy()
        uncertainty = uncertainty[0].cpu().float().numpy()

        color_predicted_source_student = colorize_mask(predicted_source_student[0].cpu().long().numpy())
        color_predicted_target_ema = colorize_mask(predicted_target_ema[0].cpu().long().numpy())
        color_predicted_mix_student = colorize_mask(predicted_mix_student[0].cpu().long().numpy())

        fig, axes = plt.subplots(
            3, 4, figsize=(int(sourceds_image.shape[1] / 100 * 4),
                           int(sourceds_image.shape[0] / 100 * 3))
        )

        axes[0][0].imshow(sourceds_image)
        axes[0][0].axis("off")
        axes[0][1].imshow(color_sourceds_target)
        axes[0][1].axis("off")
        axes[0][2].imshow(color_predicted_source_student)
        axes[0][2].axis("off")

        axes[1][0].imshow(targetds_image)
        axes[1][0].axis("off")
        axes[1][1].imshow(color_targetds_target)
        axes[1][1].axis("off")
        axes[1][2].imshow(color_predicted_target_ema)
        axes[1][2].axis("off")
        axes[1][3].imshow(uncertainty, cmap='gray', vmin=0, vmax=1)
        axes[1][3].axis("off")

        axes[2][0].imshow(mixed_img)
        axes[2][0].axis("off")
        axes[2][1].imshow(color_mixed_lbl)
        axes[2][1].axis("off")
        axes[2][2].imshow(color_predicted_mix_student)
        axes[2][2].axis("off")
        axes[2][3].imshow(mixed_seg_weight, cmap='gray', vmin=0, vmax=1)
        axes[2][3].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)

        buf.seek(0)
        concatenated_image = Image.open(buf).convert('RGB')
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"train_debug": [
                    wandb.Image(concatenated_image, file_type="jpg")
                ]
            }
        )

    @torch.no_grad()
    def _log_pred(self, image, prediction, target, dataloader_idx, pred_idx, log_prefix):
        color_ground_truth_mask = colorize_mask(target)
        color_predicted_mask = colorize_mask(prediction)

        fig, axes = plt.subplots(
            1, 3, figsize=(int(target.shape[1] / 100 * 3), target.shape[0] / 100)
        )

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(color_predicted_mask)
        axes[1].axis("off")
        axes[2].imshow(color_ground_truth_mask)
        axes[2].axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)

        buf.seek(0)
        concatenated_image = Image.open(buf).convert('RGB')
        ds_name = self.val_ds_names[dataloader_idx]
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"{log_prefix}_{ds_name}_pred_{pred_idx}": [
                    wandb.Image(concatenated_image, file_type="jpg")
                ]
            }
        )

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_crop, b_preds, b_count_mat, b_coords, b_target = batch

        all_segmentation = []
        for crop, preds, count_mat, coords, target in zip(b_crop, b_preds, b_count_mat, b_coords, b_target):
            crop = normalize(crop.float() / 255)
            segmentation = slide_inference_pre_cropped(self.network, crop, preds, count_mat, coords, self.num_classes)
            segmentation = torch.argmax(segmentation, dim=0)
            all_segmentation += [segmentation]

        for segmentation, target in zip(all_segmentation, b_target):
            self.metrics[dataloader_idx].update(
                segmentation.unsqueeze(0), target.unsqueeze(0)
            )

        if batch_idx == 0:
            for i, (img, segmentation, target) in enumerate(zip(b_image, all_segmentation, b_target)):
                # print(image.shape, segmentation.shape, target.shape)
                # torch.Size([3, 1024, 2048]) torch.Size([1024, 2048]) torch.Size([1024, 2048])

                if i < 4:  # limit images for viz
                    pred_idx = batch_idx * self.batch_size + i
                    self._log_pred(
                        # normalize_inverse(img).cpu().permute(1, 2, 0).float().numpy(),
                        img.cpu().permute(1, 2, 0).numpy(),
                        segmentation.cpu().numpy(),
                        target.cpu().numpy(),
                        dataloader_idx,
                        pred_idx,
                        log_prefix,
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
        miou_per_dataset = []
        iou_per_dataset_per_class = []
        for metric_idx, metric in enumerate(self.metrics):
            iou_per_dataset_per_class.append(metric.compute())
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]

            for iou_idx, iou in enumerate(iou_per_dataset_per_class[-1]):
                label_name = self.label2name[iou_idx]
                self.log(
                    f"{log_prefix}_{ds_name}_iou_{label_name}", iou, sync_dist=True
                )

            miou_per_dataset.append(float(iou_per_dataset_per_class[-1].mean()))
            self.log(
                f"{log_prefix}_{ds_name}_miou", miou_per_dataset[-1], sync_dist=True
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

        scaled_lr = (
                self.lr
                * math.sqrt(self.batch_size * self.trainer.num_devices * self.trainer.num_nodes)
        )

        lr = scaled_lr

        param_defs, current_params = process_parameters(
            self.network.param_defs_decoder, current_params
        )
        param_groups = [get_param_group(param_defs, lr)]

        if 0 < self.fd_weight_source or 0 < self.fd_weight_target:
            param_groups.append({"params": list(self.network_feature_head.parameters()), "lr": lr})

        lr *= self.lr_multiplier

        if 0 < len(self.network.param_defs_encoder_blocks):
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

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay, lr=scaled_lr)

        lr_scheduler = {
            "scheduler": WarmupAndLinearScheduler(
                optimizer,
                start_warmup_lr=1e-3,
                warmup_iters=self.warmup_iters,
                base_lr=1,
                final_lr=self.final_lr,
                total_iters=self.trainer.max_steps,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
