import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from datetime import datetime
import os
from config import cfg


class CrossViewiNATBirds(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset["images"])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.val = val
        if not val:
            self.transform_ground = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.ground.img_size),
                    transforms.TrivialAugmentWide(),
                    transforms.RandomHorizontalFlip(cfg.pretrain.ground.hori_flip_prob),
                    transforms.RandomVerticalFlip(cfg.pretrain.ground.vert_flip_prob),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.transform_overhead = transforms.Compose(
                [
                    transforms.RandomCrop(cfg.pretrain.overhead.img_size),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(
                        cfg.pretrain.overhead.hori_flip_prob
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform_ground = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.ground.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.transform_overhead = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.overhead.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]["file_name"]
        lat = self.images[idx]["latitude"]
        lon = self.images[idx]["longitude"]
        date = self.images[idx]["date"].split(" ")[0]
        month = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%m"))
        day = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%d"))
        date_encode = torch.tensor(
            [
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                np.sin(2 * np.pi * day / 31),
                np.cos(2 * np.pi * day / 31),
            ]
        )
        img_ground = Image.open(os.path.join("data", img_path))
        img_ground = self.transform_ground(img_ground)
        if cfg.pretrain.train.model_type == "MAE":
            return img_ground
        if not self.val:
            img_overhead = Image.open(f"data/train_overhead/images_sentinel/{idx}.jpeg")
        else:
            img_overhead = Image.open(f"data/val_overhead/images_sentinel/{idx}.jpeg")
        img_overhead = self.transform_overhead(img_overhead)
        if cfg.pretrain.train.mode == "no_metadata":
            return img_ground, img_overhead
        else:
            return (
                img_ground,
                img_overhead,
                torch.tensor(
                    [
                        np.sin(np.pi * lat / 90),
                        np.cos(np.pi * lat / 90),
                        np.sin(np.pi * lon / 180),
                        np.cos(np.pi * lon / 180),
                    ]
                ).float(),
                date_encode.float(),
            )


class CrossViewiNATBirdsFineTune(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset["images"])
        self.labels = np.array(self.dataset["categories"])
        self.species = {}
        for i in range(len(self.labels)):
            self.species[self.labels[i]["id"]] = i
        self.categories = np.array(self.dataset["annotations"])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.categories = self.categories[self.idx]
        self.val = val
        if not val:
            self.transform_ground = transforms.Compose(
                [
                    transforms.Resize(
                        cfg.pretrain.ground.img_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandAugment(
                        10, 12, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(
                        cfg.finetrain.ground.hori_flip_prob
                    ),
                    transforms.RandomVerticalFlip(cfg.finetrain.ground.vert_flip_prob),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.transform_overhead = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.overhead.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform_ground = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.ground.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.transform_overhead = transforms.Compose(
                [
                    transforms.Resize(cfg.pretrain.overhead.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]["file_name"]
        label = self.species[self.categories[idx]["category_id"]]
        lat = self.images[idx]["latitude"]
        lon = self.images[idx]["longitude"]
        date = self.images[idx]["date"].split(" ")[0]
        month = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%m"))
        day = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%d"))
        date_encode = torch.tensor(
            [
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                np.sin(2 * np.pi * day / 31),
                np.cos(2 * np.pi * day / 31),
            ]
        )
        img_ground = Image.open(os.path.join("data", img_path))
        img_ground = self.transform_ground(img_ground)
        if cfg.finetune.train.model_type == "CVEMAE" and not cfg.retrieval.enabled:
            if cfg.finetune.train.mode == "no_metadata":
                return img_ground, torch.tensor(label)
            else:
                return (
                    img_ground,
                    torch.tensor(label),
                    torch.tensor(
                        [
                            np.sin(np.pi * lat / 90),
                            np.cos(np.pi * lat / 90),
                            np.sin(np.pi * lon / 180),
                            np.cos(np.pi * lon / 180),
                        ]
                    ).float(),
                    date_encode.float(),
                )
        else:
            if not self.val:
                img_overhead = Image.open(
                    f"data/train_overhead/images_sentinel/{idx}.jpeg"
                )
            else:
                img_overhead = Image.open(
                    f"data/val_overhead/images_sentinel/{idx}.jpeg"
                )
            img_overhead = self.transform_overhead(img_overhead)
            if cfg.finetune.train.mode == "no_metadata":
                return img_ground, img_overhead, torch.tensor(label)
            else:
                return (
                    img_ground,
                    img_overhead,
                    torch.tensor(label),
                    torch.tensor(
                        [
                            np.sin(np.pi * lat / 90),
                            np.cos(np.pi * lat / 90),
                            np.sin(np.pi * lon / 180),
                            np.cos(np.pi * lon / 180),
                        ]
                    ).float(),
                    date_encode.float(),
                )


class CUBBirds(Dataset):
    def __init__(self, path, val=False):
        self.path = path
        self.images = np.loadtxt(os.path.join(self.path, "train_test_split.txt"))
        if not val:
            self.images = self.images[self.images[:, 1] == 1]
        else:
            self.images = self.images[self.images[:, 1] == 0]
        self.img_paths = np.genfromtxt(
            os.path.join(self.path, "images.txt"), dtype="str"
        )
        if not val:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (384, 384), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandAugment(
                        10, 12, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (384, 384), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.path, "images/" + self.img_paths[int(self.images[idx, 0]) - 1, 1]
        )
        label = int(self.img_paths[int(self.images[idx, 0]) - 1, 1][:3]) - 1
        img = Image.open(img_path)
        if len(np.array(img).shape) == 2:
            img_path = os.path.join(
                self.path,
                "images/" + self.img_paths[int(self.images[idx - 1, 0]) - 1, 1],
            )
            label = int(self.img_paths[int(self.images[idx - 1, 0]) - 1, 1][:3]) - 1
            img = Image.open(img_path)
        img = self.transform(img)
        return img, torch.tensor(label)


class HeirarchicalRet(Dataset):
    def __init__(self, dataset, label, ids):
        self.dataset = dataset
        self.images = np.array(self.dataset["images"])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.ids = ids
        self.transform_ground = transforms.Compose(
            [
                transforms.Resize(cfg.pretrain.ground.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_overhead = transforms.Compose(
            [
                transforms.Resize(cfg.pretrain.overhead.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self.images[self.ids[idx]]["file_name"]
        lat = self.images[self.ids[idx]]["latitude"]
        lon = self.images[self.ids[idx]]["longitude"]
        date = self.images[self.ids[idx]]["date"].split(" ")[0]
        month = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%m"))
        day = int(datetime.strptime(date, "%Y-%m-%d").date().strftime("%d"))
        date_encode = torch.tensor(
            [
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                np.sin(2 * np.pi * day / 31),
                np.cos(2 * np.pi * day / 31),
            ]
        )
        img_ground = Image.open(os.path.join("data", img_path))
        img_ground = self.transform_ground(img_ground)
        img_overhead = Image.open(
            f"data/val_overhead/images_sentinel/{self.ids[idx]}.jpeg"
        )
        img_overhead = self.transform_overhead(img_overhead)
        if cfg.pretrain.train.mode == "no_metadata":
            return img_ground, img_overhead
        else:
            return (
                img_ground,
                img_overhead,
                torch.tensor(
                    [
                        np.sin(np.pi * lat / 90),
                        np.cos(np.pi * lat / 90),
                        np.sin(np.pi * lon / 180),
                        np.cos(np.pi * lon / 180),
                    ]
                ).float(),
                date_encode.float(),
            )
