# This code is based on DeiT:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import json
import os
import time
import typing as typ
from pathlib import Path

import numpy as np
import torch
import torch as th
import torch.backends.cudnn as cudnn
from timm.data import Mixup  # type: ignore[import]
from timm.loss import LabelSmoothingCrossEntropy  # type: ignore[import]
from timm.loss import SoftTargetCrossEntropy  # type: ignore[import]
from timm.models import create_model  # type: ignore[import]
from timm.optim import create_optimizer_v2 as create_optimizer  # type: ignore[import]
from timm.optim import optimizer_kwargs
from timm.scheduler import create_scheduler  # type: ignore[import]
from timm.utils import get_state_dict  # type: ignore[import]
from timm.utils import ModelEma  # type: ignore[import]
from timm.utils import NativeScaler  # type: ignore[import]
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from augment import new_data_aug_generator
from datasets import build_dataset
from engine import evaluate
from engine import train_one_epoch
from losses import DistillationLoss
from models.resMoE import Gate
from models.resMoE import GateMoE
from samplers import RASampler
from scheduler import CurriculumScheduler
from utils import TensorboardXTracker

# import models_v2


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
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
    parser.add_argument("--debug", action="store_true")

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.3,
        metavar="PCT",
        help="Color jitter factor (default: 0.3)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
    parser.set_defaults(train_mode=True)

    parser.add_argument("--ThreeAugment", action="store_true")  # 3augment

    parser.add_argument("--src", action="store_true")  # simple random crop

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Distillation parameters
    parser.add_argument(
        "--teacher-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "regnety_160"',
    )
    parser.add_argument("--teacher-path", type=str, default="")
    parser.add_argument(
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--attn-only", action="store_true")

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
        choices=["CIFAR100", "CIFAR10", "IMNET", "IMNET100", "INAT", "INAT19"],
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
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--threshold-scheduler",
        default="cosine",
        choices=[
            "linear",
            "cosine",
        ],
        type=str,
        help="threshold scheduler",
    )
    parser.add_argument(
        "--threshold-warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup threshold, if scheduler supports",
    )
    parser.add_argument(
        "--starting-threshold",
        default=None,
        type=float,
        help="starting token skip threshold (for both attn and moe gates)",
    )
    parser.add_argument(
        "--target-threshold",
        default=None,
        type=float,
        help="target token skip threshold (for both attn and moe gates)",
    )

    # token skipping parameters
    parser.add_argument(
        "--starting-threshold-moe",
        default=1.0,
        type=float,
        help="starting token skip threshold (for both attn and moe gates)",
    )
    parser.add_argument(
        "--target-threshold-moe",
        default=0.9,
        type=float,
        help="target token skip threshold (for both attn and moe gates)",
    )

    # token skipping parameters
    parser.add_argument(
        "--starting-threshold-dense",
        default=1.0,
        type=float,
        help="starting token skip threshold (for both attn and moe gates)",
    )
    parser.add_argument(
        "--target-threshold-dense",
        default=0.9,
        type=float,
        help="target token skip threshold (for both attn and moe gates)",
    )

    parser.add_argument(
        "--gate-epoch-starting-offset",
        default=10,
        type=float,
        help="num epoch apart in which gate will start to train",
    )
    parser.add_argument(
        "--gate-epoch-ending-offset",
        default=0,
        type=float,
        help="num epoch apart in which gate will start to train",
    )
    parser.add_argument("--vis-enabled", action="store_true")
    parser.add_argument(
        "--gate-lr", default=None, type=float, help="[DEPRECATED] use --lr instead"
    )

    parser.add_argument(
        "--eval-threshold-step",
        default=0.05,
        type=float,
        help="steps when evaluating model at different thresholds",
    )

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    timestr = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    output_dir = os.path.join(args.output_dir, timestr)
    os.makedirs(output_dir, exist_ok=True)
    if args.output_dir:
        writer = TensorboardXTracker(output_dir)
    output_dir = Path(args.output_dir)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
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
        starting_threshold_dense=args.starting_threshold_dense,
        target_threshold_dense=args.target_threshold_dense,
        starting_threshold_moe=args.starting_threshold_moe,
        target_threshold_moe=args.target_threshold_moe,
    )

    curriculum_scheduler = CurriculumScheduler(args, writer)

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    if args.attn_only:
        for name_p, p in model.named_parameters():
            if ".attn." in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print("no position encoding")
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print("no patch embed")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # meanportant to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    def param_group_fn(model: torch.nn.Module):
        base_params = []
        gate_params = []
        for name, param in model.named_parameters():
            if "moe_gate" in name or "dense_gate" in name:
                gate_params.append(param)
            else:
                base_params.append(param)
        ret = [
            {"params": base_params},
            {"params": gate_params, "lr": args.lr},
        ]
        return ret

    optimizer = create_optimizer(
        model_without_ddp, **optimizer_kwargs(args), param_group_fn=param_group_fn
    )

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_path, "need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion,
        teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            if "curriculum" in checkpoint:
                curriculum_scheduler.load_state_dict(checkpoint["curriculum"])
                curriculum_scheduler.step(args.start_epoch, model)
            lr_scheduler.step(args.start_epoch)

    vis = utils.TokenSkipVisualizer(
        model=model,
        device=device,
        dataset=dataset_val,
        num_samples=8,
        writer=writer,
        args=args,
        skip_tk_brightness=0.4,  # skip tokens will be 40% as bright as non-skip
        version=int(args.model[-1]) if args.model[-1].isdigit() else 1,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    threshold = {}
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )

        curriculum_scheduler.step(epoch, model)

        lr_scheduler.step(epoch)
        writer.log_scalar("train/lr/all", optimizer.param_groups[0]["lr"], epoch)
        writer.log_scalar("train/lr/gate", optimizer.param_groups[1]["lr"], epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                        "curriculum": curriculum_scheduler.state_dict(),
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device, args)

        writer.log_scalar(
            "cuda/mem", torch.cuda.max_memory_allocated() / 1024.0**2, epoch
        )

        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

        writer.log_scalar("train/loss", train_stats["loss"], epoch)
        writer.log_scalar("test/acc1", test_stats["acc1"], epoch)

        if "loss_attn" in train_stats:
            writer.log_scalar("train/loss_attn", train_stats["loss_attn"], epoch)

        if max_accuracy < test_stats["acc1"]:
            # writer.add_scalar("Accuracy/test_acc1", test_stats["acc1"], epoch)
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / "best_checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "model_ema": get_state_dict(model_ema),
                            "scaler": loss_scaler.state_dict(),
                            "args": args,
                            "curriculum": curriculum_scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
            if args.vis_enabled:
                vis.savefig(epoch)

        print(f"Max accuracy: {max_accuracy:.2f}%")
        writer.log_scalar("test/acc1/max", max_accuracy, epoch)

        for name, m in model.named_modules():
            if isinstance(m, (Gate, GateMoE)):
                skip_ratio = m._skipped_tokens / m._total_tokens
                writer.log_scalar(f"skip_ratio/{name}", skip_ratio, epoch)
                # writer.log_scalar(f"grad-{name}", train_stats[name], epoch)
                m._skipped_tokens = 0.0
                m._total_tokens = 0.0

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    eval_threshold_list = np.linspace(
        start=args.starting_threshold,
        stop=args.target_threshold,
        num=(args.target_threshold_dense - args.starting_threshold_dense)
        / args.eval_threshold_step
        + 1,
        endpoint=True,
    )
    for name, module in model_without_ddp.named_modules():
        for current_thresh in eval_threshold_list:
            if name.endswith("dense_gate"):
                module.step(current_thresh)
            elif name.endswith("moe_gate"):
                module.step(current_thresh)
            torch.cuda.reset_peak_memory_stats()

        test_stats = evaluate(data_loader_val, model, device)
        writer.log_scalar(f"test_threshold/{current_thresh}", test_stats["acc1"], epoch)

        print(
            f"Accuracy of the network at {current_thresh} threshold on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

        current_thresh -= args.eval_threshold_step

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    vis.savefig(args.epochs, save_to_file=True)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if (args.starting_threshold is not None) and (args.target_threshold is not None):
        args.starting_threshold_dense = args.starting_threshold
        args.target_threshold_dense = args.target_threshold
        args.starting_threshold_moe = args.starting_threshold
        args.target_threshold_moe = args.target_threshold

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
