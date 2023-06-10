# pylint: disable=E1101
# mypy: disable-error-code=attr-defined
import argparse
import math
import os
import typing as typ
from typing import Callable
from typing import Tuple

import fmoe_cuda as fmoe_native
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
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
from .models.vit_moe import CustomizedGshardGate
from .models.vit_moe import CustomizedMoEMLP
from .models.vit_moe import CustomizedNaiveGate

# from tome.patch.timm import make_tome_class


class HubMeDrop(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--p-start", default=0.4, type=float)
        parser.add_argument("--p-end", default=1.0, type=float)
        parser.add_argument("--bipartite-merge", default=16, type=int)
        parser.add_argument("--step-size", default=10, type=int)
        parser.add_argument("--merge-size", default=2, type=int)
        parser.add_argument("--local-drop", action="store_true")
        # parser.add_argument("--top-k", default=2, type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tome_p = np.linspace(
            self.args.p_start, self.args.p_end, self.args.step_size
        ).tolist()
        self.tome_r = np.arange(
            0, self.args.bipartite_merge, self.args.merge_size
        ).tolist()
        #### add in initial test ####
        if self.args.p_start > 0:
            self.tome_p = [0] + self.tome_p

        self.init()

    def main(self):
        # evaluate(self.testloader, self.model, self.device)

        acc_b4, acc_af, speed = self.eval(self.tome_r, self.tome_p)
        plot_and_save(
            acc_b4,
            speed,
            self.tome_r,
            self.tome_p,
            "accuracy",
            "ms/step",
            "acc_b4 vs. speed",
            f"{self.args.output_dir}/acc_b4.pdf",
        )
        # plot(
        # acc_af,
        # speed,
        # self.tome_r,
        # "accuracy",
        # "ms/step",
        # "acc_af vs. speed",
        # f"{self.args.output_dir}/acc_af.pdf",
        # )

    def init(self):
        apply_patch(self.model, local_drop=self.args.local_drop)
        for name, param in self.model.named_parameters():
            if not "norm" in name:
                if "cls_token" in name:
                    pass
                else:
                    param.requires_grad = False

    def eval(self, tome_r: typ.List[int], tome_p):
        acc_b4 = th.zeros(len(tome_p), len(tome_r))
        acc_af = th.zeros(len(tome_p), len(tome_r))
        speed = th.zeros(len(tome_p), len(tome_r))

        # org_pth = os.path.join(self.args.output_dir, "orig.pth")
        # th.save(self.model.state_dict(), org_pth)
        self.model.k = 0
        for j, p in enumerate(tome_p):
            # __import__("pdb").set_trace()
            self.model.p = p
            for i, r in enumerate(tome_r):
                # self.model.load_state_dict(th.load(org_pth))
                self.model.r = r
                self.optimizer.__init__(self.model.parameters(), self.args.lr)
                print(f"##################### {p=} | {r=} ##################")
                throughput = self.benchmark(self.testloader)
                results_b4 = evaluate(self.testloader, self.model, self.device)
                print("fine-tune now")
                # self.finetune()
                print("fine-tunecompleted")
                # results_af = evaluate(self.testloader, self.model, self.device)
                acc_b4[j, i] = results_b4["acc1"]
                # acc_af[i] = results_af["acc1"]
                speed[j, i] = throughput["step_time"]

        # os.remove(org_pth)

        return acc_b4, acc_af, speed

    def benchmark(self, loader):
        bench = InferenceBenchmarkRunner(
            model_name=self.args.model,
            model_object=self.model,
            data_loader=loader,
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
            linestyle="--",
            label=f"hub:{labels[run_idx]}",
            lw=1,
            markersize=2,
        )

        # Annotate the points with the z labels
        # for i, label in enumerate(z):
        # plt.annotate(
        # label,
        # (x[run_idx, i], y[run_idx, i]),
        # textcoords="offset points",
        # xytext=(0, 5),
        # ha="center",
        # )

    # Set labels for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", color="gray", alpha=0.5)

    # Set the title of the plot
    plt.title(title)

    # Save the plot to a file
    plt.savefig(filename)

    # Close the plot to release memory
    plt.close()


def apply_patch(
    model, trace_source: bool = False, prop_attn: bool = True, local_drop: bool = False
):
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
    model.p = 0
    model._tome_info = {
        "r": model.r,
        "k": model.k,
        "p": model.p,
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
            if isinstance(module.mlp, CustomizedMoEMLP):
                module.mlp.gate.__class__ = GshardGateDropout
                module.mlp.gate._tome_info = module._tome_info
                module.mlp.gate.local = local_drop
                module.mlp.gate.init()

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
    if k == 0:
        return bipartite_soft_matching(metric, r, class_token, distill_token)

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

        if t < k or t // (k - 1) == 1:
            return x[:, ::2, :], x[:, 1::2, :]
        else:

            # start = t % k
            start = 0  # t % k
            step = math.ceil((t - start) / k)
            # step = (t - start) // k
            if step == 1:
                return x[:, ::2, :], x[:, 1::2, :]

            # step = max((t - 1) // (_k - 1), 1)
            # step = math.floor(t / k)

            a = x[:, start::step, :]
            # b_mask = th.ones(n, t)
            b_mask = th.ones(t).long()
            # b_mask[:, start::step] = 0
            b_mask[start::step] = 0
            b = x[:, b_mask.bool(), :]
            # swap size
            a, b = (b, a) if a.size(1) > b.size(1) else (a, b)
            return a, b

    # offset = class_token + distill_token
    # metric = metric[:, 1::]
    # metric[:, 1] = -math.inf

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # a, b = split(metric[:, offset::])
        a, b = split(metric[:, protected:])
        # print(f"a={a.shape}, b={b.shape}")

        scores = a @ b.transpose(-1, -2)
        # scores[scores.isnan()] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # print(f"node_max={node_max.shape}, node_idx={node_idx.shape}")
        # print(f"edge_idx={edge_idx.shape}")
        r = min(r, scores.shape[1], scores.shape[2])
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        # if class_token:
        # # Sort to ensure the class token is at the startjjj
        # unm_idx = unm_idx.sort(dim=1)[0]
        # print(unm_idx[0])
        # __import__("pdb").set_trace()

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x[:, protected:])  # src, dst = x[..., ::2, :], x[..., 1::2, :]

        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # if distill_token:
        # return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        # else:
        return torch.cat([x[:, 0:protected], unm, dst], dim=1)

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
        if r > 0:
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


class NaiveGateDropout(CustomizedNaiveGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        p = self._tome_info["p"].pop()
        gate_top_k_idx = self.token_drop(self, gate_score, gate_top_k_idx, p)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def token_drop(self, gate_score, topk_idx, p):
        if p == 0:
            return topk_idx

        B, Tk, _ = self.original_shape

        num_dropped = int(p * Tk)

        prob_dist = gate_score
        uni_dist = th.full_like(gate_score, 1 / gate_score.size(-1))
        jsd = JSD()
        score = jsd(prob_dist, uni_dist)
        score = score.view(B, Tk, -1)
        # topk_idx = topk_idx.view(B, Tk, -1)
        sorted_idx = th.argsort(score, dim=1)[:, 0:num_dropped].squeeze()
        rows = th.arange(B).view(-1, 1).expand_as(sorted_idx).to(score.device)
        sorted_idx = (Tk * rows + sorted_idx).view(-1)
        topk_idx[sorted_idx] = -1
        return topk_idx


class GshardGateDropout(CustomizedGshardGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def naive_out(self, inp, return_all_scores=False):
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def init(self):
        self.jsd = JSD()

    def forward(self, x):

        naive_outs = self.naive_out(x, True)
        topk_idx, topk_val, gate_score = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
            )
            / S
        )
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert**2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // (self.world_size * self.num_expert)
        capacity = (
            torch.ones(
                self.num_expert * self.world_size,
                dtype=torch.int32,
                device=topk_idx.device,
            )
            * capacity
        )

        topk_idx = fmoe_native.prune_gate_by_capacity(
            topk_idx, capacity, self.num_expert, self.world_size
        )

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = 2 * topk_val[:, 1] < rand_routing_prob
            topk_idx[:, 1].masked_fill_(mask, -1)

        p = self._tome_info["p"].pop()
        topk_idx = self.token_drop(gate_score, topk_idx, p)

        return topk_idx, topk_val

    def token_drop(self, gate_score, topk_idx, p):
        if p == 0:
            return topk_idx

        prob_dist = gate_score.softmax(dim=-1)
        uni_dist = th.full_like(gate_score, 1 / gate_score.size(-1))
        score = self.jsd(prob_dist, uni_dist)

        if self.local:
            B, Tk, _ = self.original_shape
            num_dropped = int(p * Tk)
            score = score.view(B, Tk, -1)
            sorted_idx = th.argsort(score, dim=1)[:, 0:num_dropped].squeeze()
            rows = th.arange(B).view(-1, 1).expand_as(sorted_idx).to(score.device)
            sorted_idx = (Tk * rows + sorted_idx).view(-1)
        else:
            num_dropped = int(p * topk_idx.size(0))
            sorted_idx = th.argsort(score, dim=0)[0:num_dropped].squeeze()

        topk_idx[sorted_idx] = -1
        return topk_idx


def make_tome_class(transformer_class):
    class HubMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["k"] = parse_r(len(self.blocks), self.k)
            self._tome_info["p"] = [self.p] * len(self.blocks)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return HubMeVisionTransformer


def alpha_D(D, n1: int, n2: int):
    return 2 * (-D.square() * 2 * n1 / (1 + n1 / n2)).exp()


@torch.jit.script
def kolmogorov_smirnov(
    points1, points2, alpha=torch.as_tensor([0.05, 0.01, 0.001, 0.0001])
):
    """
    Kolmogorov-Smirnov test for empirical similarity of probability distributions.

    Warning: we assume that none of the elements of points1 coincide with points2.
    The test may gave false negatives if there are coincidences, however the effect
    is small.
    Parameters
    ----------
    points1 : (..., n1) torch.Tensor
        Batched set of samples from the first distribution
    points2 : (..., n2) torch.Tensor
        Batched set of samples from the second distribution
    alpha : torch.Tensor
        Confidence intervals we wish to test. The default is torch.as_tensor([0.05, 0.01, 0.001, 0.0001]).
    Returns
    -------
    Tuple of (torch.Tensor, torch.Tensor)
        The test result at each alpha, and the estimated p-values.
    """
    device = points1.device
    n1 = points1.shape[-1]
    n2 = points2.shape[-1]
    # Confidence level
    c_ks = torch.sqrt(-0.5 * (alpha / 2).log())
    sup_conf = c_ks * torch.as_tensor((n1 + n2) / (n1 * n2)).sqrt()
    sup_conf = sup_conf.reshape((1, alpha.shape[0]))
    sup_conf = sup_conf.to(device)

    comb = torch.concatenate((points1, points2), dim=-1)

    comb_argsort = comb.argsort(dim=-1)

    pdf1 = torch.where(comb_argsort < n1, 1 / n1, 0)
    pdf2 = torch.where(comb_argsort >= n1, 1 / n2, 0)

    cdf1 = pdf1.cumsum(dim=-1)
    cdf2 = pdf2.cumsum(dim=-1)

    sup, _ = (cdf1 - cdf2).abs().max(dim=-1, keepdim=True)
    return sup > sup_conf, alpha_D(sup, n1, n2)


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m).sum(dim=-1) + self.kl(q.log(), m).sum(dim=-1))
