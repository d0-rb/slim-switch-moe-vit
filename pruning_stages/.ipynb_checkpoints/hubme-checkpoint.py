# pylint: disable=E1101
# mypy: disable-error-code=attr-defined
import argparse
import math
import os
import typing as typ
from typing import Callable
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
from tome.merge import bipartite_soft_matching
from tome.merge import merge_source
from tome.merge import merge_wavg
from tome.patch.timm import ToMeAttention
from tome.patch.timm import ToMeBlock
from tome.utils import parse_r

from .base import BasePruning
from .benchmark import InferenceBenchmarkRunner
from .engine import evaluate
from .engine import train_one_epoch
from .models.vision_transformer import Attention
from .models.vision_transformer import Block

# from tome.patch.timm import make_tome_class


class HubMeDrop(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        # parser.add_argument("--attn-momentum", default=0.75, type=float)
        parser.add_argument("--finetune-epochs", default=10, type=int)
        # parser.add_argument("--top-k", default=2, type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.tome_r = np.arange(0, 24, 4)
        # self.tome_k = np.arange(28, 98j, 7)
        self.tome_r = np.arange(0, 24, 12)
        # self.tome_k = np.arange(28, 98, 7)
        # self.tome_r = [15]
        self.tome_k = [28]
        self.init()

    def main(self):
        # evaluate(self.testloader, self.model, self.device)

        acc_b4, acc_af, speed = self.eval(self.tome_r, self.tome_k)
        plot_and_save(
            acc_b4,
            speed,
            self.tome_r,
            self.tome_k,
            "accuracy",
            "ms/step",
            "acc_b4 vs. speed",
            f"{self.args.output_dir}/acc_b4.pdf",
        )
        plot(
            acc_af,
            speed,
            self.tome_r,
            "accuracy",
            "ms/step",
            "acc_af vs. speed",
            f"{self.args.output_dir}/acc_af.pdf",
        )

    def init(self):
        apply_patch(self.model)
        for name, param in self.model.named_parameters():
            if not "norm" in name:
                if "cls_token" in name:
                    pass
                else:
                    param.requires_grad = False

    def eval(self, tome_r: typ.List[int], tome_k):
        acc_b4 = th.zeros(len(tome_k), len(tome_r))
        acc_af = th.zeros(len(tome_k), len(tome_r))
        speed = th.zeros(len(tome_k), len(tome_r))

        org_pth = os.path.join(self.args.output_dir, "orig.pth")
        th.save(self.model.state_dict(), org_pth)
        for j, k in enumerate(tome_k):
            self.model.k = k
            for i, r in enumerate(tome_r):
                self.model.load_state_dict(th.load(org_pth))
                self.model.r = r
                self.optimizer.__init__(self.model.parameters(), self.args.lr)
                print(f"##################### {k=} | {r=} ##################")
                throughput = self.benchmark()
                results_b4 = evaluate(self.testloader, self.model, self.device)
                print("fine-tune now")
                # self.finetune()
                print("fine-tunecompleted")
                # results_af = evaluate(self.testloader, self.model, self.device)
                acc_b4[j, i] = results_b4["acc1"]
                # acc_af[i] = results_af["acc1"]
                speed[j, i] = throughput["step_time"]

        os.remove(org_pth)

        return acc_b4, acc_af, speed

    def benchmark(self):
        bench = InferenceBenchmarkRunner(
            model_name=self.args.model,
            model_object=self.model,
            **vars(self.args),
        )
        results = bench.run()
        print(
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step"
        )
        return results

    def finetune(self):
        best_stats = None
        for epoch in range(self.args.finetune_epochs):
            th.cuda.reset_peak_memory_stats()
            if self.args.distributed:
                self.valloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.valloader,
                self.optimizer,
                self.device,
                epoch,
                self.loss_scaler,
                self.args.clip_grad,
                None,
                self.mixup_fn,
                set_training_mode=self.args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                args=self.args,
            )
            test_stats = evaluate(self.testloader, self.model, self.device)
            if best_stats is None or best_stats["acc1"] < test_stats["acc1"]:
                th.save(
                    self.model.state_dict(),
                    os.path.join(self.args.output_dir, "temp_best_ckpt.pth"),
                )
                best_stats = test_stats
        self.model.load_state_dict(
            th.load(os.path.join(self.args.output_dir, "temp_best_ckpt.pth"))
        )
        os.remove(os.path.join(self.args.output_dir, "temp_best_ckpt.pth"))


def plot(x, y, z, xlabel, ylabel, title, filename):
    # Create the line plot
    plt.clf()
    plt.plot(x, y, marker="o", linestyle="-")

    # Annotate the points with the z labels
    for i, label in enumerate(z):
        plt.annotate(
            label, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center"
        )

    # Set labels for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the title of the plot
    plt.title(title)

    # Save the plot to a file
    plt.savefig(filename)

    # Close the plot to release memory
    plt.close()


def plot_and_save(x, y, z, labels, xlabel, ylabel, title, filename):
    plt.clf()
    # Ensure input data is NumPy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Get the number of runs
    num_runs = x.shape[0]

    # Create the line plot for each run
    for run_idx in range(num_runs):
        plt.plot(
            x[run_idx],
            y[run_idx],
            marker="o",
            linestyle="-",
            label=f"hub:{labels[run_idx]}",
        )

        # Annotate the points with the z labels
        for i, label in enumerate(z):
            plt.annotate(
                label,
                (x[run_idx, i], y[run_idx, i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )

    # Set labels for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the title of the plot
    plt.title(title)

    # Save the plot to a file
    plt.savefig(filename)

    # Close the plot to release memory
    plt.close()


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.
    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.k = 0
    model._tome_info = {
        "r": model.r,
        "k": model.k,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = HubMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = HubMeAttention


def do_nothing(x, mode=None):
    return x


def bipartite_cls_matching(
    metric: torch.Tensor,
    r: int,
    k: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    # if class_attn is None:
    # return bipartite_soft_matching(metric, r, class_token, distill_token)

    # class_attn.shape (B x T)

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    # def split(x):
    # n, t, c = x.shape
    # cls_idx = class_attn.argsort(dim=-1, descending=True)[..., None]
    # # cls_idx.shape B x T_idx
    # a_idx = cls_idx[:, : t // 2]
    # b_idx = cls_idx[:, t // 2 : :]
    # a = x.gather(dim=-2, index=cls_idx[:, : t // 2].expand(n, t // 2, c))
    # b = x.gather(
    # dim=-2, index=cls_idx[:, t // 2 :].expand(n, t - a_idx.shape[1], c)
    # )
    # return a, b
    def split(x):
        n, t, c = x.shape
        # step = t // k
        step = math.floor(t / k)

        hub = x[:, ::step, :]
        b_mask = th.ones(n, t)
        # b_mask = th.ones(t).long()
        b_mask[:, ::step] = 0
        # b_mask[::step] = 0
        # unm = x[:, b_mask, :]
        b_idx = th.where(b_mask)[1].view(n, -1, 1).to(x.device)
        unm = x.gather(dim=-2, index=b_idx.expand(-1, -1, c))
        return hub, unm

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if class_token:
            metric[:, 0] = -math.inf
        if distill_token:
            metric[:, 1] = -math.inf

        a, b = split(metric)
        # print(f"a={a.shape}, b={b.shape}")

        scores = a @ b.transpose(-1, -2)
        scores[scores.isnan()] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # print(f"node_max={node_max.shape}, node_idx={node_idx.shape}")
        # print(f"edge_idx={edge_idx.shape}")

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)  # src, dst = x[..., ::2, :], x[..., 1::2, :]

        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


class HubMeBlock(ToMeBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        k = self._tome_info["k"].pop(0)
        if r > 0 and k > 0:
            # Apply ToMe here
            merge, _ = bipartite_cls_matching(
                metric,
                r,
                k,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class HubMeAttention(ToMeAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        # mean_cls_attn = attn[:, :, 0, :].mean(dim=1)
        # mean_cls_attn = attn.
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class HubMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["k"] = parse_r(len(self.blocks), self.k)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return HubMeVisionTransformer
