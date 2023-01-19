from __future__ import annotations

import math
import typing as typ

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from fmoe import FMoETransformerMLP  # type: ignore[import]
from timm.models import register_model  # type: ignore[import]
from torch.autograd import Variable

from .model import DistilledVisionTransformer as Deit
from .resMoE import CustomizedMoEMLP
from .resMoE import Gate
from .resMoE import GateMoE
from .vision_transformer import Block


class GateImnet(GateMoE):
    def __init__(self, norm_layer: typ.Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm_layer

    def forward(self, x, attn):
        # assume x is sorted by attn
        threshold = self._threshold  # if self.training else self.threshold
        density = int(x.size(1) * threshold)  # type: ignore[operator]

        tokens = x[:, 0:density]
        self.tk_idx = th.arange(x.size(1))
        self.gate_attn = attn

        skip_tokens = None
        summary_token = None
        if x.size(1) - density > 0:
            skip_tokens = x[:, density::]

            values = attn[:, density::].clone()
            values = values.unsqueeze(dim=-1)

            summary_skip_token = self.norm(
                (skip_tokens * values).sum(dim=1, keepdim=True)
            )
            self._skipped_tokens += math.prod(values.shape)
        else:
            summary_skip_token = None
            self._skipped_tokens = 0.0

        self._total_tokens += math.prod(x.shape[0:2])

        return tokens, skip_tokens, summary_skip_token


def index_select(x, index):
    """index_select code donated by Junru."""
    B, T, D = x.shape
    index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
    return th.gather(input=x, dim=1, index=index_repeat)


def forward_block_vanilla(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def forward_block_w_attn(self, x):
    x = self._forward(self, x)
    attn = self.attn.x_cls_attn.sum(dim=1)
    patch_tk, attn_patch = cls_attn_reordering(x[:, 1::], attn[:, 1::])
    return th.cat((x[:, 0].unsqueeze(dim=1), patch_tk), dim=1), attn_patch


def forward_block_no_attn(self, input_):
    x, attn = self._forward(self, input_)
    return x


def cls_attn_reordering(patch_tk: th.Tensor, cls_patch_attn: th.Tensor):
    # cls_path_attn (B x T) from B x H x N x N
    cls_path_attn_sorted, index = cls_patch_attn.sort(dim=1, descending=True)
    patch_tk_sorted = index_select(patch_tk, index)
    return patch_tk_sorted, cls_path_attn_sorted


def forward_residule_moe(
    self, input_: typ.Tuple[th.Tensor, th.Tensor]
):  # x, class_attn):

    x, class_attn = input_
    x = self.norm1(x)

    cls_token, patch_tk = x[:, 0], x[:, 1::]
    cls_token = cls_token.unsqueeze(dim=1)
    # patch_tk, prev_attn = cls_attn_reordering(patch_tk, class_attn)

    tokens, skip_tk_attn, summary_skip_token = self.dense_gate(
        patch_tk, class_attn
    )  # , 1::])

    tokens_to_attn = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [
                    cls_token,
                    tokens,
                    summary_skip_token,
                ],
            )
        ),
        dim=1,
    )

    attended_tokens = self.drop_path(self.attn(tokens_to_attn))

    cur_attn = self.attn.x_cls_attn.sum(dim=1)[:, 1::]  # B x patch_tk_size()

    if summary_skip_token is not None:
        attn_skip_token = attended_tokens[:, -1].unsqueeze(dim=1)
        attended_tokens = attended_tokens[:, 0:-1]

        token_attn = cur_attn[:, 0:-1]
        skip_attn = cur_attn[:, -1]

        class_attn[:, 0 : token_attn.size(1)] = (
            class_attn[:, 0 : token_attn.size(1)] + token_attn
        )
        class_attn[:, token_attn.size(1) : :] = class_attn[
            :, token_attn.size(1) : :
        ] + skip_attn.unsqueeze(dim=-1)
        class_attn /= 2.0

    else:
        attn_skip_token = None
        class_attn = (class_attn + cur_attn) / 2.0
        skip_attn = None

    attended_tokens = attended_tokens + tokens_to_attn[:, 0 : attended_tokens.size(1)]

    attended_tokens = self.norm2(attended_tokens)

    #######################
    # attended_cls_token = attended_tokens[:, 0]
    attended_patch_tk = attended_tokens[:, 1::]
    attended_patch_tk, attended_patch_attention = cls_attn_reordering(
        attended_patch_tk, class_attn[:, 0 : attended_patch_tk.size(1)]
    )
    # patch_tk = attended_tokens[:, patch_idx::]
    tokens, skip_tk_mlp, summary_skip_token = self.moe_gate(
        attended_patch_tk, attended_patch_attention
    )  # , 1print("#2 tokens:", tokens.shape)

    tokens_to_fwd = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [
                    attended_tokens[:, 0].unsqueeze(dim=1),
                    tokens,
                    summary_skip_token,
                    attn_skip_token,
                ],
            )
        ),
        dim=1,
    )

    fwded_tokens = self.drop_path(self.mlp(tokens_to_fwd))

    if attn_skip_token is not None:
        summary_skip_token_mlp_full = fwded_tokens[:, -1].unsqueeze(dim=1)
        fwded_tokens = fwded_tokens[:, 0:-1]
    else:
        summary_skip_token_mlp_full = None

    if summary_skip_token is not None:
        summary_skip_token_mlp_half = fwded_tokens[:, -1].unsqueeze(dim=1)
        fwded_tokens = fwded_tokens[:, 0:-1]
    else:
        summary_skip_token_mlp_half = None

    fwded_tokens += tokens_to_fwd[:, 0 : fwded_tokens.size(1)]

    if skip_tk_mlp is not None:
        update_skip_tk_2 = skip_tk_mlp + summary_skip_token_mlp_half.tile(
            skip_tk_mlp.size(1)
        ).view(skip_tk_mlp.shape)

        fwded_tokens = th.cat((fwded_tokens, update_skip_tk_2), dim=1)

    if skip_tk_attn is not None:  # and summary_skip_token is not None:
        update_skip_tk_1 = (
            skip_tk_attn
            + attn_skip_token.tile(skip_tk_attn.size(1)).view(skip_tk_attn.shape)
            + summary_skip_token_mlp_full.tile(skip_tk_attn.size(1)).view(
                skip_tk_attn.shape
            )
        )

        fwded_tokens = th.cat((fwded_tokens, update_skip_tk_1), dim=1)

    class_attn = th.cat(
        (attended_patch_attention, class_attn[:, attended_patch_attention.size(1) : :]),
        dim=1,
    )

    patch_tk, class_attn = cls_attn_reordering(fwded_tokens[:, 1::], class_attn)

    return th.cat((fwded_tokens[:, 0].unsqueeze(dim=1), patch_tk), dim=1), class_attn


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_base_patch16_224
from .model import deit_tiny_distilled_patch16_224


@register_model
def resmoe_tiny_patch16_224_expert8_imnet_v4(
    pretrained=False,
    starting_threshold_dense=1.0,
    target_threshold_dense=0.9,
    starting_threshold_moe=1.0,
    target_threshold_moe=0.9,
    **kwargs,
):
    model = deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4
    drop_rate = 0.0

    moe_placement = [0, 1] * (depth // 2)
    index = 0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            is_moe = moe_placement.pop(0)
            if index == 0:
                print(f"{name=}")
                index += 1
                bound_method = forward_block_w_attn.__get__(module, module.__class__)
                module._forward = forward_block_vanilla
                setattr(module, "forward", bound_method)

            else:
                module.dense_gate = GateImnet(
                    module.norm1,
                    module.attn,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    is_clk_tk=True,
                    is_dist_tk=False,
                )

                module.moe_gate = GateImnet(
                    module.norm2,
                    module.attn,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    is_clk_tk=True,
                    is_dist_tk=False,
                )
                if is_moe:

                    module.mlp = CustomizedMoEMLP(
                        embed_dim,
                        embed_dim * mlp_ratio,
                        moe_num_experts=8,
                        moe_top_k=2,
                        drop=drop_rate,
                    )
                module.is_cls_token = True
                module.is_dist_token = False
                if index < depth - 1:
                    print(f"{name=} type 2")
                    bound_method = forward_residule_moe.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                else:
                    print(f"{name=} type 3")
                    module._forward = forward_residule_moe
                    bound_method = forward_block_no_attn.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                index += 1
    return model
