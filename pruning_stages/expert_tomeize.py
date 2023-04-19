import argparse

from .base import BasePruning


class ExpertTomeize(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--tome-keep-rate", default=1.0, type=float, help="ToMe keep rate (1.0 = no merging)")
        parser.add_argument("--tome-every", default=3, type=int, help="How often to apply ToMe (every n blocks)")
        parser.add_argument("--tome-delay", default=0, type=int, help="How many blocks to delay ToMe by (0 = no delay)")

    def __init__(self, model, args, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tome_keep_rate = args.tome_keep_rate
        self.tome_every = args.tome_every
        self.tome_delay = args.tome_delay

        if self.tome_keep_rate < 0.5:
            raise ValueError('ToMe keep rate must be >= 0.5')
    
    def main(self):
        for i, block in enumerate(self.model.blocks.children(), start=-self.tome_delay)[self.tome_delay:]:
            if i % self.tome_every == 0:
                self.tomeize(block)
    
    def tomeize(self_tomeize, block):
        def new_block_forward(self, x):
            cls_exists = self.cls_token is not None
            dist_exists = self.dist_token is not None

            x = x + self.drop_path(self.attn(self.norm1(x)))

            # ToMe
            b, t, d= x.shape
            keep_tokens = int((t - cls_exists - dist_exists) * self_tomeize.tome_keep_rate)

            merge, _ = bipartite_soft_matching(x, keep_tokens, cls_exists, dist_exists)
            x = merge(x)
            # end ToMe

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        
        bound_forward = new_block_forward.__get__(block, block.__class__)
        setattr(block, 'forward', bound_forward)

# code taken from https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch as th


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: th.Tensor,
    r: int,
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

    with th.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: th.Tensor, mode="mean") -> th.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return th.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return th.cat([unm, dst], dim=1)

    def unmerge(x: th.Tensor) -> th.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = th.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge
