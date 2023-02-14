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

    def set_expert_mapping(self, mapping: th.Tensor):
        """mapping: 1D array ex: [0,1,2,3,4,5] which maps expert at index position to expert at
        value position"""
        assert mapping.shape == self.expert_mapping.shape
        assert th.max(mapping) < self.tot_expert and th.min(mapping) >= 0
        self.expert_mapping.data.copy_(mapping.data)

    @staticmethod
    def apply_expert_mapping(self, inputs, output):
        # [0,1,2,3,4,5,6]
        # [1,1,1,0,5,0,0]
        # output[0] = Tokes x topk
        # [[3,2], [5, 6]] -> [[0,1], [0,0]]
        # -1 -> skipped processing
        return self.expert_mapping[output[0]], *output[1::]


class CustomizedGshardGate(GShardGate):
    """fmoe's naive gate with experts mapping"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("expert_mapping", th.arange(self.tot_expert))
        self.register_forward_hook(self.apply_expert_mapping)

    def set_expert_mapping(self, mapping: th.Tensor):
        """mapping: 1D array ex: [0,1,2,3,4,5] which maps expert at index position to expert at
        value position"""
        assert mapping.shape == self.expert_mapping.shape
        assert th.max(mapping) < self.tot_expert and th.min(mapping) >= 0
        self.expert_mapping.data.copy_(mapping.data)

    @staticmethod
    def apply_expert_mapping(self, inputs, output):
        return self.expert_mapping[output[0]], *output[1::]


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_base_patch16_224


def _make_moe(model, settings, pretrained=False):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, Block):
            if cnt % 2 == 0:
                fc1 = module.mlp.fc1
                fc2 = module.mlp.fc2

                module.mlp = CustomizedMoEMLP(
                    settings["embed_dim"],
                    settings["embed_dim"] * settings["mlp_ratio"],
                    moe_num_experts=settings["num_experts"],
                    moe_top_k=2,
                    drop=settings["drop_rate"],
                    gate=settings["gate"],
                )
                if pretrained:
                    with th.no_grad():
                        module.mlp.experts.htoh4.weight.data.copy_(
                            fc1.weight.tile(settings["num_experts"], 1, 1)
                        )
                        module.mlp.experts.htoh4.bias.data.copy_(
                            fc1.bias.tile(settings["num_experts"], 1)
                        )
                        module.mlp.experts.h4toh.weight.data.copy_(
                            fc2.weight.tile(settings["num_experts"], 1, 1)
                        )
                        module.mlp.experts.h4toh.bias.data.copy_(
                            fc2.bias.tile(settings["num_experts"], 1)
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
