#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import torch
import torchvision
from os.path import join
from torch.utils.data import get_worker_info
from typing import Union

from datasets.loader.in1k_zip_dataset import IN1KZipDataset
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule


class ImageNet1kDataModule(CustomLightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            img_size: int,
            train_num_workers: int,
            val_num_workers: int = 6,
            val_batch_size: int = 4,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            img_size=img_size,
            train_num_workers=train_num_workers,
        )
        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size

        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.RandomResizedCrop(self.img_size),
            torchvision.transforms.ToTensor(),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.CenterCrop(self.img_size),
            torchvision.transforms.ToTensor(),
        ])

        # self.train_dataset = torchvision.datasets.ImageNet(self.root, split='train', transform=train_transform)
        # self.val_dataset = torchvision.datasets.ImageNet(self.root, split='val', transform=val_transform)

        self.train_dataset = IN1KZipDataset(join(self.root, "in1k_train.zip"), transform=train_transform)
        self.val_dataset = IN1KZipDataset(join(self.root, "in1k_val.zip"), transform=val_transform)

        print("Train ds sizes:", len(self.train_dataset))
        print("Val ds sizes:", len(self.val_dataset))

        return self

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            num_workers=self.train_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            persistent_workers=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.val_batch_size,
        )
