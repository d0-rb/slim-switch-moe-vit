# This code is based on DeiT:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import os
import time
import typing as typ
import matplotlib
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from timm.models import create_model  # type: ignore[import]
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from models.resMoE import Gate
import matplotlib.pyplot as plt

import utils
from datasets import build_dataset
from engine import evaluate

# import models_v2

GATE_NAMES = ('dense_gate', 'moe_gate')  # names of gates IN THE ORDER OF WHICH THEY ARE PROCESSED


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--bce-loss", action="store_true")
    parser.add_argument("--unscale-lr", action="store_true")

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR100", "CIFAR10", "IMNET", "INAT", "INAT19"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # visualiztion parameters
    parser.add_argument(
        "--gate-depth", default=0, type=int, help="depth of gate to visualize"
    )
    parser.add_argument(
        "--gate-name", default="dense_gate", type=str, help="name of gate to visualize (dense_gate or moe_gate)", choices=GATE_NAMES,
    )

    return parser


def main(args):
    # utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    timestr = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    output_dir = os.path.join(args.output_dir, timestr)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # switch to evaluation mode
    model.eval()

    negative_mean = [-channel_mean for channel_mean in IMAGENET_DEFAULT_MEAN]
    inverse_std = [1/channel_std for channel_std in IMAGENET_DEFAULT_STD]
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=inverse_std),
        transforms.Normalize(mean=negative_mean, std=[1., 1., 1.]),
    ])
    indices = []  # will contain index mappings to be composed

    images, target = next(iter(data_loader_val))

    display_img = unnormalize(images).cpu().numpy()
    display_img = np.transpose(display_img, (0, 2, 3, 1))

    patch_size = model.patch_embed.patch_size
    grid_size = model.patch_embed.grid_size

    images = images.to(device, non_blocking=True)

    def add_idx(gate, _x, output):
        nonlocal indices
        
        indices.append(gate.tk_idx.detach().cpu().numpy())

    def visualize_gate_output(gate, _x, output):
        # x.shape (B, Tokens, dim)
        # tk_idx.shape (B, Tokens) [first n tokens along dim 1 are selected, rest are skip]

        x = _x[0]  # get only positional argument
        B, T, D = x.shape

        nonlocal display_img
        nonlocal indices
        
        indices.append(gate.tk_idx.detach().cpu().numpy())
        n = int(x.size(1) * gate.threshold)  # number of selected tokens

        total_idx = np.repeat(np.expand_dims(np.arange(T), 0), B, axis=0)  # final idx mapping made by composing everything in indexes
        for index in indices:  # composing all index mappings
            total_idx = np.take_along_axis(total_idx, index, axis=1)

        sel_idx = total_idx[:, :n]  # indices of selected tokens
        tk_mask = np.full((B, T), 0.4)
        np.put_along_axis(tk_mask, sel_idx, np.ones_like(sel_idx), axis=1)  # np equivalent of torch.scatter to go from indices to mask
        
        tk_mask = np.reshape(tk_mask, (B, *grid_size, 1))  # tk_mask.shape (B, H_patch, W_patch, 1)
        img_mask = np.kron(tk_mask, np.ones((1, *patch_size, 3)))  # img_mask.shape (B, H, W, 3)

        display_img = img_mask * display_img
        display_img = np.concatenate(display_img, axis=1)  # display_img.shape (H, W * B, 3)
        plt.imshow(display_img)
        plt.savefig(output_dir / 'fig.png')


    for depth in range(args.gate_depth):  # track all gates up to current gate in previous blocks
        for gate_name in GATE_NAMES:
            current_gate = getattr(model.blocks[depth], gate_name)

            if not isinstance(current_gate, Gate):
                print(f'invalid gate {gate_name} at block {depth}')
                return

            current_gate.register_forward_hook(add_idx)

    for gate_name in GATE_NAMES:  # track all gates up to current gate in current block
        current_gate = getattr(model.blocks[args.gate_depth], gate_name)

        if args.gate_name == gate_name:
            current_gate.register_forward_hook(visualize_gate_output)
            break
        else:
            current_gate.register_forward_hook(add_idx)

    # compute output
    with torch.cuda.amp.autocast():
        output = model(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "ResMoE gate skip visualization script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
