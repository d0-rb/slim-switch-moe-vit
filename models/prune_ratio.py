import collections
import copy
import math
import typing as typ
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.utils.prune as prune
import tqdm  # type: ignore[import]


# from layers import Conv2d
# from layers import Linear

__all__ = [
    "generate_mask_parameters",
    "SynFlow",
    "Mag",
    "Taylor1ScorerAbs",
    "Rand",
    "SNIP",
    "GraSP",
    "check_sparsity",
    "check_sparsity_dict",
    "prune_model_identity",
    "prune_model_custom",
    "extract_mask",
    "ERK",
    "PHEW",
    "Ramanujan",
    "rigl_initialization_",
]


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask."""
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def generate_mask_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "blocks" in name:
            mask = th.one_like(module.weight)
            yield mask, module.weight, name


class Pruner:
    def __init__(self, masked_parameters: typ.Iterator[typ.List[th.Tensor]]):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}  # type: ignore[var-annotated]

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally."""
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = th.cat([th.flatten(v) for v in self.scores.values()])
        k = int((1 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold = global_scores.topk(k)[0][-1]
            for mask, param, name in self.masked_parameters:
                score = self.scores[id(param)]
                zero = th.tensor([0.0]).to(mask.device)
                one = th.tensor([1.0]).to(mask.device)
                mask.copy_(th.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise."""
        for mask, param, name in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = th.kthvalue(th.flatten(score), k)
                zero = th.tensor([0.0]).to(mask.device)
                one = th.tensor([1.0]).to(mask.device)
                mask.copy_(th.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope."""
        if scope == "global":
            self._global_mask(sparsity)
        if scope == "local":
            self._local_mask(sparsity)

    @th.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters."""
        for mask, param, name in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, _ in self.masked_parameters:
            shape = mask.shape
            perm = th.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters."""
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p, n in self.masked_parameters:
            self.scores[id(p)] = th.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = th.cat([th.flatten(v) for v in self.scores.values()])
        norm = th.sum(all_scores)
        for _, p, n in self.masked_parameters:
            self.scores[id(p)].div_(norm)

    def density_per_block(self):
        total = sum(m.sum() for m, _, _ in self.masked_parameters)
        block_n = 0
        block_density = {}
        for mask, _, name in self.masked_parameters:
