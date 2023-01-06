# This code is based on DeiT:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import io
import os
import time
from collections import defaultdict
from collections import deque
from collections.abc import Sequence
import numpy as np
import itertools

import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD

from models.resMoE import Gate


# https://gitlab.com/prakhark2/relevance-mapping-networks/-/blob/master/code/dataload.py
def get_split_cifar100(opt, task_id, class_size=5, shuffle=False):
    start_class = (task_id - 1) * class_size
    end_class = task_id * class_size

    if opt.dataset == "s-cifar100":
        opt.dataset = "cifar100"
    train, test = data_create(opt)
    targets_train = torch.tensor(train.targets)
    target_train_idx = (targets_train >= start_class) & (targets_train < end_class)

    targets_test = torch.tensor(test.targets)
    target_test_idx = (targets_test >= start_class) & (targets_test < end_class)

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True,
    )  # check this
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        drop_last=True,
    )  # check this
    if opt.dataset == "cifar100":
        opt.dataset = "s-cifar100"
    return trainloader, testloader


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({"state_dict_ema": checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class TensorboardXTracker:
    def __init__(self, log_dir):
        self.writer = tensorboardX.SummaryWriter(log_dir)

    def log_scalar(self, var_name, value, step):
        self.writer.add_scalar(var_name, value, step)

    def log_loss(self, loss, step):
        self.log_scalar("loss", loss, step)

    def log_validation_acc(self, acc, step):
        self.log_scalar("validation_acc", acc, step)

    def log_test_acc(self, acc, step):
        self.log_scalar("test_acc", acc, step)

    def add_image(self, var_name, img, step):
        self.writer.add_image(var_name, img, step)
    
    def add_tk_skp_vis(self, depth, gate, img, step):
        self.add_image(f'block_{depth}_{gate}', img, step)

    def close(self):
        self.writer.close()


class TokenSkipVisualizer:
    def __init__(
        self,
        model: nn.Module,
        device,
        dataset,
        num_samples: int,
        writer,
        args,
        skip_tk_brightness: float = 0.4,  # skip tokens will be 40% as bright as non-skip
    ):
        global_rank = get_rank()

        if global_rank != 0:
            return
        
        sampler = torch.utils.data.RandomSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=num_samples,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        
        self.model = model.module if args.distributed else model

        negative_mean = [-channel_mean for channel_mean in IMAGENET_DEFAULT_MEAN]
        inverse_std = [1/channel_std for channel_std in IMAGENET_DEFAULT_STD]
        self.unnormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=inverse_std),
            transforms.Normalize(mean=negative_mean, std=[1., 1., 1.]),
        ])
        self.indices = []  # will contain index mappings to be composed, should be reset before every visualization
        self.vis_gates = []
        self.writer = writer
        self.skip_tk_brightness = skip_tk_brightness
        self.step = 0  # for tensorboard
        self.track_idx = False  # for disabling/enabling vis hooks

        images, target = next(iter(dataloader))

        self.display_img = self.unnormalize(images).cpu()  # original images for reference

        self.patch_size = self.model.patch_embed.patch_size
        self.grid_size = self.model.patch_embed.grid_size

        self.images = images.to(device, non_blocking=True)


        for depth, block in enumerate(self.model.blocks):
            for gate_name, gate in block.named_children():
                if not isinstance(gate, Gate):
                    continue

                self.vis_gates.append((depth, gate_name))
                vis_hook = self._idx_vis_hook(depth, gate_name)
                gate.register_forward_hook(vis_hook)
    

    def _idx_vis_hook(self, depth, name):
        gate_tuple = (depth, name)

        def gate_hook(gate, _x, output):
            if not self.track_idx:
                return

            self.indices.append(gate.tk_idx.detach().cpu())
            
            if not gate_tuple in self.vis_gates:
                return

            # if we are to output a visualization for this layer
            x = _x[0]  # get only positional argument

            # x.shape (B, Tokens, dim)
            B, T, D = x.shape
            
            n = int(x.size(1) * gate._threshold)  # number of selected tokens

            # total_idx.shape (B, Tokens) [first n tokens along dim 1 are selected, rest are skip]
            total_idx = torch.arange(T, device='cpu').unsqueeze(0).repeat((B, 1))  # final idx mapping made by composing everything in indexes
            for index in self.indices:  # composing all index mappings
                total_idx = torch.gather(total_idx, dim=1, index=index)

            sel_idx = total_idx[:, :n]  # indices of selected tokens
            tk_mask = torch.full((B, T), self.skip_tk_brightness, dtype=torch.float)
            tk_mask.scatter_(dim=1, index=sel_idx, src=torch.ones_like(sel_idx, dtype=torch.float))  # np equivalent of torch.scatter to go from indices to mask
            
            tk_mask = tk_mask.view(B, 1, *self.grid_size)  # tk_mask.shape (B, H_patch, W_patch, 1)
            img_mask = torch.kron(tk_mask, torch.ones((1, 3, *self.patch_size)))  # img_mask.shape (B, 3, H, W)

            masked_img = img_mask * self.display_img
            masked_img = make_grid(masked_img)

            self.writer.add_tk_skp_vis(depth, name, masked_img, self.step)
            
        return gate_hook
    
    @property
    def track_idx(self):
        return getattr(self, '_track_idx', False)
        
    @track_idx.setter
    def track_idx(self, track_idx):
        self._track_idx = track_idx

        if track_idx:
            self.indices = []


    def savefig(self, step, gates: None | Sequence[tuple[int, str]] = None):
        """
        save visualization of token skipping at current state of network on given gates

        :param gates: Sequence of 2-tuples representing a gate, e.g. [(0, 'dense_gate'), (2, 'moe_gate')] or None to use stored gates to output
        """
        global_rank = get_rank()

        if global_rank != 0:
            return
        
        if gates is not None:
            self.vis_gates = gates
        
        self.step = step
        self.track_idx = True

        # compute output
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.model(self.images)
        
        self.track_idx = False

