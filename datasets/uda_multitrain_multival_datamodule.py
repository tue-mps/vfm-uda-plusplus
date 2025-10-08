#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import torch
from pathlib import Path
from torch.utils.data import get_worker_info
from typing import Union, List

import datasets
from datasets.loader.zip_dataset import ZipDataset, ImageOnlyZipDataset
from datasets.rare_class_sampler_dataset import RareClassSamplerDataset
from datasets.transform.inference.resize_and_crop_transform import ResizeAndCropTransform
from datasets.transform.train.uda_transform import UDATransform
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule
from datasets.utils.mappings import get_cityscapes_mapping, get_mapillary_mapping, get_synthia_mapping, \
    get_urbansyn_mapping
from datasets.utils.source_target_dataset2 import SourceTargetDataset2


class UDAMultiTrainMultiValDataModule(CustomLightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            train_num_workers: int,
            val_num_workers: int = 6,
            in_img_scale: float = 1.0,
            img_size: int = 1024,
            scale_gta5: List[float] = [0.9, 1.1],
            scale_synscapes: List[float] = [0.9, 1.1],
            scale_cityscapes: List[float] = [0.9, 1.1],
            scale_bdd100k: List[float] = [0.9, 1.1],
            scale_mapillary: List[float] = [0.9, 1.1],
            scale_wilddash: List[float] = [0.9, 1.1],
            scale_acdc: List[float] = [0.9, 1.1],
            scale_darkzurich: List[float] = [0.9, 1.1],
            scale_synthia: List[float] = [0.9, 1.1],
            use_rcs: bool = True,
            rcs_class_temp: float = 0.05,
            cat_max_ratio: float = 0.75,
            sources: list = None,
            targets: list = None,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            img_size=img_size,
            train_num_workers=train_num_workers,
        )
        self.val_num_workers = val_num_workers
        self.rcs_class_temp = rcs_class_temp
        self.cat_max_ratio = cat_max_ratio
        self.use_rcs = use_rcs
        self.val_batch_size = 1
        self.sources = sources if sources is not None else []
        self.targets = targets if targets is not None else []
        # assert self.val_batch_size == 1  # val with multi ds requires batch size 1
        assert not bool(set(self.sources) & set(self.targets)), "sources and targets shouldn't share any datasets"
        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

        self.gta5_train_transforms = UDATransform(
            img_size=int(1440 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_gta5[0],
            max_scale=scale_gta5[1],
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("gta5" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("gta5" in self.sources and use_rcs)
        )

        self.synscapes_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_synscapes[0],
            max_scale=scale_synscapes[1],
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("synscapes" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("synscapes" in self.sources and use_rcs)
        )

        self.cityscapes_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_cityscapes[0],
            max_scale=scale_cityscapes[1],
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("cityscapes" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("cityscapes" in self.sources and use_rcs)
        )

        self.bdd100k_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_bdd100k[0],
            max_scale=scale_bdd100k[1],
            cat_max_ratio=cat_max_ratio if ("bdd" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("bdd" in self.sources and use_rcs)
        )
        self.mapillary_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_mapillary[0],
            max_scale=scale_mapillary[1],
            label_mapping=get_mapillary_mapping(),
            cat_max_ratio=cat_max_ratio if ("mapillary" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("mapillary" in self.sources and use_rcs)
        )

        self.wilddash_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_wilddash[0],
            max_scale=scale_wilddash[1],
            label_mapping=get_cityscapes_mapping(),
            cat_max_ratio=cat_max_ratio if ("wilddash" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("wilddash" in self.sources and use_rcs)
        )

        self.acdc_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_acdc[0],
            max_scale=scale_acdc[1],
            cat_max_ratio=cat_max_ratio if ("acdc" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("acdc" in self.sources and use_rcs)
        )

        self.darkzurich_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_darkzurich[0],
            max_scale=scale_darkzurich[1],
            cat_max_ratio=cat_max_ratio if ("darkzurich" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("darkzurich" in self.sources and use_rcs)
        )

        self.synthia_train_transforms = UDATransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size,
            min_scale=scale_synthia[0],
            max_scale=scale_synthia[1],
            label_mapping=get_synthia_mapping(),
            cat_max_ratio=cat_max_ratio if ("synthia" in self.sources) and (not use_rcs) else 1.0,
            do_cropping=not ("synthia" in self.sources and use_rcs)
        )

        self.cityscapes_val_transforms = ResizeAndCropTransform(
            label_mapping=get_cityscapes_mapping(),
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )
        # 720x1280
        self.bdd100k_val_transforms = ResizeAndCropTransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )
        # 1080 × 1920
        self.mapillary_val_transforms = ResizeAndCropTransform(
            label_mapping=get_mapillary_mapping(),
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )

        # 1080 × 1920
        self.wilddash_val_transforms = ResizeAndCropTransform(
            label_mapping=get_cityscapes_mapping(),
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )

        self.acdc_val_transforms = ResizeAndCropTransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )

        self.darkzurich_val_transforms = ResizeAndCropTransform(
            img_size=int(1024 * in_img_scale),
            crop_size=img_size
        )


    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        gta5_train_datasets = [
            ZipDataset(
                zip_path=Path(self.root, f"{i:02}_images.zip"),
                target_zip_path=Path(self.root, f"{i:02}_labels.zip"),
                transforms=self.gta5_train_transforms,
                image_folder_path_in_zip=Path("./images"),
                target_folder_path_in_zip=Path("./labels"),
                image_suffix=".png",
                target_suffix=".png",
            )
            for i in range(1, 11)
        ]
        self.gta5_train_dataset = torch.utils.data.ConcatDataset(gta5_train_datasets)

        self.synscapes_train_dataset = ZipDataset(
            transforms=self.synscapes_train_transforms,
            image_folder_path_in_zip=Path("./Synscapes/img/rgb-2k"),
            target_folder_path_in_zip=Path("./Synscapes/img/class"),
            image_suffix=".png",
            target_suffix=".png",
            zip_path=Path(
                self.root,
                "synscapes.zip",
            ),
        )

        self.cityscapes_train_dataset = ZipDataset(
            transforms=self.cityscapes_train_transforms,
            image_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            image_suffix=".png",
            target_suffix=".png",
            image_stem_suffix="leftImg8bit",
            target_stem_suffix="gtFine_labelIds",
            zip_path=Path(self.root, "leftImg8bit_trainvaltest.zip"),
            target_zip_path=Path(self.root, "gtFine_trainvaltest.zip"),
        )

        self.bdd100k_train_dataset = ZipDataset(
            transforms=self.bdd100k_train_transforms,
            image_folder_path_in_zip=Path("./bdd100k/images/10k/train"),
            target_folder_path_in_zip=Path("./bdd100k/labels/sem_seg/masks/train"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "bdd100k_images_10k.zip"),
            target_zip_path=Path(self.root, "bdd100k_sem_seg_labels_trainval.zip"),
        )

        self.mapillary_train_dataset = ZipDataset(
            transforms=self.mapillary_train_transforms,
            image_folder_path_in_zip=Path("./training/images"),
            target_folder_path_in_zip=Path("./training/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(
                self.root,
                "An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA.zip",
            ),
        )

        self.wilddash_train_dataset = ZipDataset(
            transforms=self.wilddash_train_transforms,
            image_folder_path_in_zip=Path("wilddash/train/images"),
            target_folder_path_in_zip=Path("wilddash/train/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "wilddash.zip", ),
        )

        self.acdc_train_dataset = ZipDataset(
            transforms=self.acdc_train_transforms,
            image_folder_path_in_zip=Path("acdc_rgb_anon_train/rgb_anon"),
            target_folder_path_in_zip=Path("acdc_gt_train/gt"),
            image_suffix=".png",
            target_suffix=".png",
            image_stem_suffix="rgb_anon",
            target_stem_suffix="gt_labelTrainIds",
            zip_path=Path(self.root, "acdc_rgb_anon_train.zip"),
            target_zip_path=Path(self.root, "acdc_gt_train.zip"),
        )

        self.darkzurich_train_dataset = ImageOnlyZipDataset(
            transforms=self.darkzurich_train_transforms,
            image_folder_path_in_zip=Path("rgb_anon/train"),
            image_suffix=".png",
            zip_path=Path(self.root, "Dark_Zurich_train_anon.zip"),
        )

        self.synthia_train_dataset = ZipDataset(
            transforms=self.synthia_train_transforms,
            image_folder_path_in_zip=Path("RAND_CITYSCAPES/RGB"),
            target_folder_path_in_zip=Path("RAND_CITYSCAPES/GT/LABELS"),
            image_suffix=".png",
            target_suffix=".png",
            zip_path=Path(self.root, "SYNTHIA_RAND_CITYSCAPES.zip"),
            is_synthia=True
        )


        self.all_train_datasets = {
            "gta5": self.gta5_train_dataset,
            "cityscapes": self.cityscapes_train_dataset,
            "bdd": self.bdd100k_train_dataset,
            "mapillary": self.mapillary_train_dataset,
            "synscapes": self.synscapes_train_dataset,
            "wilddash": self.wilddash_train_dataset,
            "acdc": self.acdc_train_dataset,
            "darkzurich": self.darkzurich_train_dataset,
            "synthia": self.synthia_train_dataset,
        }


        self.cityscapes_val_dataset = ZipDataset(
            transforms=self.cityscapes_val_transforms,
            image_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            image_suffix=".png",
            target_suffix=".png",
            image_stem_suffix="leftImg8bit",
            target_stem_suffix="gtFine_labelIds",
            zip_path=Path(self.root, "leftImg8bit_trainvaltest.zip"),
            target_zip_path=Path(self.root, "gtFine_trainvaltest.zip"),
        )

        self.bdd100k_val_dataset = ZipDataset(
            transforms=self.bdd100k_val_transforms,
            image_folder_path_in_zip=Path("./bdd100k/images/10k/val"),
            target_folder_path_in_zip=Path("./bdd100k/labels/sem_seg/masks/val"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "bdd100k_images_10k.zip"),
            target_zip_path=Path(self.root, "bdd100k_sem_seg_labels_trainval.zip"),
        )

        self.mapillary_val_dataset = ZipDataset(
            transforms=self.mapillary_val_transforms,
            image_folder_path_in_zip=Path("./validation/images"),
            target_folder_path_in_zip=Path("./validation/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(
                self.root,
                "An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA.zip",
            ),
        )

        self.wilddash_val_dataset = ZipDataset(
            transforms=self.wilddash_val_transforms,
            image_folder_path_in_zip=Path("wilddash/val/images"),
            target_folder_path_in_zip=Path("wilddash/val/labels"),
            image_suffix=".jpg",
            target_suffix=".png",
            zip_path=Path(self.root, "wilddash.zip", ),
        )

        self.acdc_val_dataset = ZipDataset(
            transforms=self.acdc_val_transforms,
            image_folder_path_in_zip=Path("acdc_rgb_anon_val/rgb_anon"),
            target_folder_path_in_zip=Path("acdc_gt_val/gt"),
            image_stem_suffix="rgb_anon",
            target_stem_suffix="gt_labelTrainIds",
            image_suffix=".png",
            target_suffix=".png",
            zip_path=Path(self.root, "acdc_rgb_anon_val.zip"),
            target_zip_path=Path(self.root, "acdc_gt_val.zip"),
        )

        self.darkzurich_val_dataset = ZipDataset(
            transforms=self.darkzurich_val_transforms,
            image_folder_path_in_zip=Path("rgb_anon/val/night"),
            target_folder_path_in_zip=Path("gt/val/night"),
            image_stem_suffix="rgb_anon",
            target_stem_suffix="gt_labelTrainIds",
            image_suffix=".png",
            target_suffix=".png",
            zip_path=Path(self.root, "Dark_Zurich_val_anon.zip"),
            target_zip_path=Path(self.root, "Dark_Zurich_val_anon.zip"),
        )

        self.all_val_datasets = {
            "cityscapes": self.cityscapes_val_dataset,
            "bdd": self.bdd100k_val_dataset,
            "mapilllary": self.mapillary_val_dataset,
            "wilddash": self.wilddash_val_dataset,
            "acdc": self.acdc_val_dataset,
            "darkzurich": self.darkzurich_val_dataset,
        }

        print("Train ds sizes:", {k: len(v) for k, v in self.all_train_datasets.items()})
        print("Val ds sizes:", {k: len(v) for k, v in self.all_val_datasets.items()})

        return self

    def train_dataloader(self):
        if self.use_rcs:
            source = torch.utils.data.ConcatDataset([
                RareClassSamplerDataset(
                    source=self.all_train_datasets[ds_name],
                    source_name=ds_name,
                    crop_size=self.img_size,
                    rcs_class_temp=self.rcs_class_temp,
                    cat_max_ratio=self.cat_max_ratio
                )
                for ds_name in self.sources
            ])
        else:
            source = torch.utils.data.ConcatDataset([
                self.all_train_datasets[ds_name] for ds_name in self.sources
            ])

        if 0 < len(self.targets):
            target = torch.utils.data.ConcatDataset([
                self.all_train_datasets[ds] for ds in self.targets
            ])

            return torch.utils.data.DataLoader(
                SourceTargetDataset2(source, target),
                shuffle=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                num_workers=self.train_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.batch_size,
            )
        else:
            return torch.utils.data.DataLoader(
                source,
                shuffle=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                num_workers=self.train_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.batch_size,
            )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                self.cityscapes_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.bdd100k_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.mapillary_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.wilddash_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.acdc_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
            torch.utils.data.DataLoader(
                self.darkzurich_val_dataset,
                persistent_workers=False,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_memory,
                batch_size=self.val_batch_size,
                collate_fn=datasets.utils.util._eval_collate_fn,
            ),
        ]
