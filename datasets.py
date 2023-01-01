# This code is based on DeiT:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

import torch

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import ImageFolder

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def data_create(opt):

    if opt.data_set == 'mnist' or 'pmnist':
        dataset = datasets.MNIST(root=opt.data_path, download=True, train=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(opt.input_size),
                                        transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10), \
                                            scale=(0.8, 1.2), translate=(0.05, 0.05))],p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        valset = datasets.MNIST(root=opt.data_path, download=True, train=False,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),
                                   ]))
        nb_classes = 10

    if opt.data_set == 'fmnist':
        dataset = datasets.FashionMNIST(root=opt.data_path, download=True, train=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(opt.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        valset = datasets.FashionMNIST(root=opt.data_path, download=True, train=False,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),
                                   ]))
        nb_classes = 10

    if opt.data_set == 'svhn':
        dataset = datasets.SVHN(root=opt.data_path, download=True, split='train',
                            transform=transforms.Compose([transforms.Resize(opt.input_size), transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        valset = datasets.SVHN(root=opt.data_path, download=True, split='test',
                           transform=transforms.Compose([transforms.Resize(opt.input_size), transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        nb_classes = 10

    elif opt.data_set in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = datasets.ImageFolder(root=opt.data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.input_size),
                                       transforms.CenterCrop(opt.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nb_classes = 1000

    elif opt.data_set == 'lsun':
        dataset = datasets.LSUN(db_path=opt.data_path, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.input_size),
                                transforms.CenterCrop(opt.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nb_classes = 30

    elif opt.data_set == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.data_path, download=True,
                               transform=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                # transforms.Lambda(lambda x: add_noise(x))
                                ]))
        valset = datasets.CIFAR10(root=opt.data_path, download=True, train=False,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                # transforms.Lambda(lambda x: add_noise(x))
                                ]))
        nb_classes = 10

    elif opt.data_set == 'cifar100' or opt.data_set == 'cifar':
        dataset = datasets.CIFAR100(root=opt.data_path, download=False,
                                transform=transforms.Compose([
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  # transforms.Lambda(lambda x: add_noise(x))
                                ]))
        valset = datasets.CIFAR100(root=opt.data_path, download=False, train=False,
                               transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                # transforms.Lambda(lambda x: add_noise(x))
                                ]))
        nb_classes = 100

    return dataset, valset, nb_classes


# https://gitlab.com/prakhark2/relevance-mapping-networks/-/blob/master/code/dataload.py
def build_split_dataset(is_train, opt, start_class, class_size=5):
    end_class = start_class + class_size

    dataset, nb_classes = build_dataset(is_train, opt)
    targets = torch.tensor(dataset.targets)
    target_idx = ((targets >= start_class) & (targets < end_class))

    subset = torch.utils.data.dataset.Subset(dataset, np.where(target_idx==1)[0])

    return subset, nb_classes


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class IMAGENET100(ImageFolder):
    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if os.path.exists("imnet100"):
            f = open("imnet100/train_classes.pkl", "rb")
            classes = pickle.load(f)
            f.close()
            f = open("imnet100/train_class_to_idx.pkl", "rb")
            class_to_idx = pickle.load(f)
            f.close()
            print("Loaded classes")
            return classes, class_to_idx
        classes = [d.name for d in os.scandir(dir) if d.is_dir()][:100]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == "CIFAR100":
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 100
    elif args.data_set == "CIFAR10":
        dataset = datasets.CIFAR10(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 10
    elif args.data_set == "CAR":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 196
    elif args.data_set == "FLOWER":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    elif args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "INAT":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2018,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "INAT19":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2019,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "IMNET100":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = IMAGENET100(root, transform=transform)
        nb_classes = 100

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
