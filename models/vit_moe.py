from __future__ import annotations

import math
import typing as typ

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from fmoe import FMoETransformerMLP  # type: ignore[import]
from fmoe.gates import NaiveGate  # type: ignore[import]
from timm.models import register_model  # type: ignore[import]
from torch.autograd import Variable

from .model import DistilledVisionTransformer as Deit
from .vision_transformer import Block


class CustomizedMoEMLP(FMoETransformerMLP):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        moe_num_experts: int,
        moe_top_k: int,
        drop: float,
        act_layer: typ.Callable = th.nn.GELU,
        **kwargs,
    ):
        activation = nn.Sequential(*[act_layer(), nn.Dropout(p=drop)])
        # use naive-gate
        super().__init__(
            moe_num_experts,
            in_features,
            hidden_features,
            activation,
            top_k=moe_top_k,
            **kwargs,
        )


class CustomizedNaiveGate(NaiveGate):
    """fmoe's naive gate with experts mapping"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("expert_mapping", th.arange(self.tot_expert))

    def forward(self, inp, return_all_scores=False):
        gate = self.apply_expert_mapping(self.gate(inp))
        gate_top_k_val, gate_top_k_idx = th.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def set_expert_mapping(self, mapping: th.Tensor):
        """mapping: 1D array ex: [0,1,2,3,4,5] which maps expert at index position to expert at
        value position"""
        assert mapping.shape == self.expert_mapping.shape
        assert th.max(mapping) < self.tot_expert and th.min(mapping) >= 0
        self.expert_mapping.data.copy_(mapping.data)

    def apply_expert_mapping(self, probability: th.Tensor):
        probability = probability[:, self.expert_mapping]
        return probability


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_base_patch16_224


@register_model
def moe_base_patch16_224(pretrained=False, **kwargs):
    model = deit_base_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4
    drop_rate = 0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=kwargs["num_experts"],
                moe_top_k=2,
                drop=drop_rate,
                gate=CustomizedNaiveGate,
            )
    __import__("pdb").set_trace()
    return model


@register_model
def moe_small_patch16_224(pretrained=False, **kwargs):
    model = deit_small_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4
    drop_rate = 0.0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=kwargs["num_experts"],
                moe_top_k=2,
                drop=drop_rate,
                gate=CustomizedNaiveGate,
            )
    return model


@register_model
def moe_tiny_patch16_224(pretrained=False, **kwargs):
    model = deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4
    drop_rate = 0.0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=kwargs["num_experts"],
                moe_top_k=2,
                drop=drop_rate,
                gate=CustomizedNaiveGate,
            )
    return model
