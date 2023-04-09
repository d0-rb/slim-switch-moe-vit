from __future__ import annotations

import math
import typing as typ

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from fmoe import FMoETransformerMLP  # type: ignore[import]
from fmoe.gates import GShardGate
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
        self.register_forward_hook(self.apply_expert_mapping)

        self.gate_scores = None

    def set_expert_mapping(self, mapping: th.Tensor):
        """mapping: 1D array ex: [0,1,2,3,4,5] which maps expert at index position to expert at
        value position"""
        assert mapping.shape == self.expert_mapping.shape
        assert th.max(mapping) < self.tot_expert and th.min(mapping) >= -1
        self.expert_mapping.data.copy_(mapping.data)

    # @staticmethod
    # def apply_expert_mapping(self, inputs, output):
    #     self.gate_score = output[1].detach().clone()
    #     return self.expert_mapping[output[0]], *output[1::]

    @staticmethod
    def apply_expert_mapping(self, inputs, output):
        # [0,1,2,3,4,5,6]
        # [1,1,1,0,5,0,0]
        # output[0] = Tokes x topk
        # [[3,2], [5, 6]] -> [[0,1], [0,0]]
        # -1 -> skipped processing
        # gate_top_k_idx = output[0]
        gate_score = output[1].clone()
        gate_top_k_idx = self.expert_mapping[output[0]]
        mask = gate_top_k_idx[:, 0] == gate_top_k_idx[:, 1]
        gate_score[mask, 1] = 0
        gate_score[mask, 0] = 1 - gate_score[mask, 0].detach() + gate_score[mask, 0]
        gate_top_k_idx[mask, 1] = -1
        if len(output) == 2:
            return gate_top_k_idx, gate_score
        else:
            return gate_top_k_idx, gate_score, output[-1]


class CustomizedGshardGate(GShardGate):
    """fmoe's naive gate with experts mapping"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("expert_mapping", th.arange(self.tot_expert))
        self.register_forward_hook(self.apply_expert_mapping)

        self.gate_scores = None

    def set_expert_mapping(self, mapping: th.Tensor):
        """mapping: 1D array ex: [0,1,2,3,4,5] which maps expert at index position to expert at
        value position"""
        assert mapping.shape == self.expert_mapping.shape
        assert th.max(mapping) < self.tot_expert and th.min(mapping) >= -1
        self.expert_mapping.data.copy_(mapping.data)

    @staticmethod
    def apply_expert_mapping(self, inputs, output):
        self.gate_score = output[1]
        return self.expert_mapping[output[0]], *output[1::]


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_base_patch16_224


def _get_weights(model, bia=True):
    weights = []
    if bia:
        bias = []
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, CustomizedMoEMLP):
            weights.append((mlp.experts.htoh4.weight.data, mlp.experts.h4toh.weight.data))
            if bia:
                bias.append((mlp.experts.htoh4.bias.data, mlp.experts.h4toh.bias.data))
    return weights, bias

def _set_weights(model, weights_new, bias_new=None):
    index = 0
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, CustomizedMoEMLP):
            htoh4, h4toh = weights_new[index]
            mlp.experts.htoh4.weight.data.copy_(htoh4)
            mlp.experts.h4toh.weight.data.copy_(h4toh)
            if bias_new is not None:
                htoh4_bias, h4toh_bias = bias_new[index]
                mlp.experts.htoh4.bias.data.copy_(htoh4_bias)
                mlp.experts.h4toh.bias.data.copy_(h4toh_bias)
            # mlp.experts.htoh4.weight.data = htoh4
            # mlp.experts.h4toh.weight.data = h4toh
            index += 1
    return model

def _set_expert_mapping(model, mapping_list):
    index = 0
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, CustomizedMoEMLP):
            # if hasattr(mlp.gate, "set_expert_mapping")
            mlp.gate.set_expert_mapping(mapping_list[index])
            index += 1
    return model



def _make_moe(model, settings, pretrained=False):
    cnt = 0
    hidden_dim = (settings["embed_dim"] * settings["mlp_ratio"])
    # hidden_dim = (settings["embed_dim"] * settings["mlp_ratio"]) // 2
    for name, module in model.named_modules():
        if isinstance(module, Block):
            if cnt % 2 == 0:
                fc1 = module.mlp.fc1
                fc2 = module.mlp.fc2

                module.mlp = CustomizedMoEMLP(
                    settings["embed_dim"],
                    hidden_dim,
                    moe_num_experts=settings["num_experts"],
                    moe_top_k=2,
                    drop=settings["drop_rate"],
                    gate=settings["gate"],
                )
            cnt += 1
    return model


@register_model
def moe_base_patch16_224(pretrained=False, gate="naive", **kwargs):
    model = deit_base_patch16_224(pretrained=pretrained, **kwargs)
    settings = {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "drop_rate": 0,
        "num_experts": kwargs.get("num_experts", 32),
        "gate": CustomizedNaiveGate if gate == "naive" else CustomizedGshardGate,
    }
    return _make_moe(model, settings, pretrained)


@register_model
def moe_small_patch16_224(pretrained=False, gate="naive", **kwargs):
    model = deit_small_patch16_224(pretrained=pretrained, **kwargs)
    settings = {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
        "drop_rate": 0.0,
        "num_experts": kwargs.get("num_experts", 32),
        "gate": CustomizedNaiveGate if gate == "naive" else CustomizedGshardGate,
    }
    return _make_moe(model, settings, pretrained)


@register_model
def moe_tiny_patch16_224(pretrained=False, gate="naive", **kwargs):
    model = deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    settings = {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4,
        "drop_rate": 0.0,
        "num_experts": kwargs.get("num_experts", 32),
        "gate": CustomizedNaiveGate if gate == "naive" else CustomizedGshardGate,
    }
    return _make_moe(model, settings, pretrained)
