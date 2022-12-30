import math
import typing as typ

import torch as th
import torch.nn as nn
from fmoe import FMoETransformerMLP  # type: ignore[import]
from timm.models import register_model  # type: ignore[import]

from .model import DistilledVisionTransformer as Deit
from .vision_transformer import Block

__all__ = ["Gate", "ResBlock", "resmoe_tiny_patch16_224_expert8"]


class CustomizedMoEMLP(FMoETransformerMLP):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        moe_num_experts: int,
        moe_top_k: int,
        drop: float,
        act_layer: typ.Callable = th.nn.GELU,
    ):
        activation = nn.Sequential(*[act_layer(), nn.Dropout(p=drop)])
        # use naive-gate
        super().__init__(
            moe_num_experts, in_features, hidden_features, activation, top_k=moe_top_k
        )


class Gate(nn.Module):
    def __init__(
        self,
        in_dim: int,
        tau: float,
        dropout: float = 0.0,
        ratio: int = 10,
    ):
        super().__init__()
        out_dim = ratio
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_dim, out_dim))
        self.tau = tau

        self._total_tokens = 0
        self._skipped_tokens = 0

    def forward(self, x):
        # x.shape (B x Tokens x dim)
        out = self.head(x)  # (B x Token x 2)
        # out is a category choice (either skip or don't), we use gumbel_softmax to relax it
        if self.training:
            choice = nn.functional.gumbel_softmax(out, self.tau, hard=True, dim=-1)
        else:
            index = out.topk(1, dim=-1)[-1]
            choice = th.scatter(th.zeros_like(out), dim=-1, index=index, value=1.0)
            # (B X token X ratio)

        self._total_tokens += math.prod(out.shape[0:2])
        self._skipped_tokens += choice[:, :, 0].sum().detach().item()

        choice[:, :, 1] = choice[:, :, 1::].sum(dim=-1)
        return choice[:, :, 0:2]


class ResBlock(Block):
    def __init__(self, *args, num_expert=32, moe_top_k=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp = CustomizedMoEMLP(
            args[0],
            args[0] * kwargs["mlp_ratio"],
            kwargs["act_layer"],
            num_expert,
            moe_top_k,
            kwargs["drop_rate"],
        )

        self.dense_gate = Gate(args[0], 1, 0.0, 10)
        self.moe_gate = Gate(args[0], 1, 0.0, 10)

    def forward(self, x):
        x = self.norm1(x)
        # x.shape (B x Tokens x dim)

        mask = self.dense_gate(x)

        skip_tk = x * mask[:, :, 0].unsqueeze(dim=-1)
        tk = x * mask[:, :, 1].unsqueeze(dim=-1)

        x = self.drop_path(self.attn(tk)) + tk + skip_tk
        x = self.norm2(x)

        mask = self.moe_gate(x)

        skip_tk = x * mask[:, :, 0].unsqueeze(dim=-1)
        tk = x * mask[:, :, 1].unsqueeze(dim=-1)

        x = self.drop_path(self.mlp(tk)) + tk + skip_tk

        return x


def forward_residule_moe(self, x):
    # x.shape (B x Tokens x dim)

    mask = self.dense_gate(x)
    x = self.norm1(x)

    skip_tk = x * mask[:, :, 0].unsqueeze(dim=-1)
    tk = x * mask[:, :, 1].unsqueeze(dim=-1)

    x = self.drop_path(self.attn(tk)) + tk + skip_tk

    mask = self.moe_gate(x)
    x = self.norm2(x)

    skip_tk = x * mask[:, :, 0].unsqueeze(dim=-1)
    tk = x * mask[:, :, 1].unsqueeze(dim=-1)

    x = self.drop_path(self.mlp(tk)) + tk + skip_tk

    return x


from .model import deit_tiny_patch16_224


@register_model
def resmoe_tiny_patch16_224_expert8(pretrained=False, **kwargs):
    model = deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4
    drop_rate = 0.0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(embed_dim, 1.0, ratio=10)
            module.moe_gate = Gate(embed_dim, 1.0, ratio=10)

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            bound_method = forward_residule_moe.__get__(module, module.__class__)
            setattr(module, "forward", bound_method)
    return model


@register_model
def moe_tiny_patch16_224_expert8(pretrained=False, **kwargs):
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
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
    return model
