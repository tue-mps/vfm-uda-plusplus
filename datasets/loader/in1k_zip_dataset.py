#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#


import json
import re
import torch
import zipfile
from PIL import Image
from pathlib import Path
from torch.utils.data import get_worker_info
from typing import Tuple


class IN1KZipDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            zip_path: str,
            transform,
    ):
        self.zip_path = zip_path
        self.transform = transform
        self.zip = {}

        image_zip = self._load_zips()

        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=self._sort_key)
        }

        with open('datasets/loader/imagenet_class_index.json', 'r') as file:
            label_json = json.load(file)
        self.classid2dirname = {int(k): v[0] for k, v in label_json.items()}
        self.dirname2clasid = {v: k for k, v in self.classid2dirname.items()}

        self.images = []
        self.targets = []
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(m):
                continue
            target_name = Path(m.filename).parent.name
            self.images.append(image_name)
            self.targets.append(self.dirname2clasid[target_name])

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)
        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
    ):
        return (m.filename.endswith("JPEG"))

    def _load_zips(self) -> zipfile.ZipFile:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)
        return self.zip[worker]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_zip = self._load_zips()

        with image_zip.open(
                self.image_folder_members[self.images[index]].filename
        ) as image_file:
            image = Image.open(image_file).convert("RGB")

        return self.transform(image), self.targets[index]

    def __del__(self):
        for o in self.zip.values():
            o.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["zip"] = {}
        return state
