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
from .vision_transformer import Block

__all__ = [
    "Gate",
    "ResBlock",
    "resmoe_tiny_patch16_224_expert8",
    "resmoe_tiny_patch16_224_expert8_attn_loss",
    "resmoe_tiny_patch16_224_expert8_attn_loss_v2",
    "Block",
    "resvit_tiny_patch16_224",
]


def sampler(tensor, tau, temperature):
    """gumbel noise addition

    :input: TODO
    :tau: TODO
    :temperature: TODO
    :returns: TODO

    """
    noise = th.rand_like(tensor)
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise)
    return (tensor + noise) / tau + temperature


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
        dropout: float = 0.5,
        target_threshold: float = 0.9,
        starting_threshold: float = 1.0,
        is_hard: float = True,
        add_guass_noise: bool = False,
        keep_prev_mask=False,
    ):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(in_dim, 1) if not keep_prev_mask else nn.Identity()
        self.register_buffer("_threshold", th.tensor(starting_threshold))
        self.register_buffer("threshold", th.tensor(target_threshold))
        self.register_buffer("tau", th.tensor(tau))

        self._total_tokens = 0
        self._skipped_tokens = 0

        self.is_hard = is_hard
        self.disable = False
        self.tk_idx = None
        self.add_guass_noise = add_guass_noise
        self.keep_prev_mask = keep_prev_mask

    def step(self, threshold: th.Tensor):
        self._threshold.data.copy_(threshold)

    def step_tau(self, delta: th.Tensor):
        tau = self.tau - delta
        self.tau.data.copy_(
            max(tau, 0.0)  # type: ignore[call-overload]
        )  # type: ignore[operator]

    def forward(
        self, x: th.Tensor
    ) -> typ.Tuple[th.Tensor, th.Tensor | None, th.Tensor | None, th.Tensor | None]:

        if self.disable:
            return x, None, None, None

        tokens: th.Tensor
        skip_tokens: th.Tensor | None = None
        summary_token: th.Tensor | None = None
        summary_skip_token: th.Tensor | None = None
        # cuda = x.device

        threshold = self._threshold  # if self.training else self.threshold
        density = int(x.size(1) * threshold)  # type: ignore[operator]

        if self.keep_prev_mask:
            logits = th.zeros(x.size(0), x.size(1)).to(x.device)
            logits[:, 0:density] = 1.0
        else:
            logits = self.head(self.dropout(x)).squeeze()  # (B x Token x 1)
            if self.training and self.add_guass_noise:
                logits = logits + th.rand_like(logits) * self.tau

        # prob = th.sigmoid(out)

        values_tk, index = logits.topk(k=density, dim=1)
        self.tk_idx = index

        tokens = self.index_select(x, index)

        if self.training:
            values = values_tk.softmax(dim=-1)
            summary_token = (tokens * values.unsqueeze(dim=-1)).sum(dim=1, keepdim=True)

        if x.size(1) - density > 0:
            values_skip_tk, index = logits.topk(
                k=x.size(1) - density, dim=1, largest=False
            )
            self.tk_idx = th.cat([self.tk_idx, index], dim=1)
            skip_tokens = self.index_select(x, index)
            # if x_cls is not None:
            values = values_skip_tk.softmax(dim=-1)
            summary_skip_token = (skip_tokens * values.unsqueeze(dim=-1)).sum(
                dim=1, keepdim=True
            )
            self.gate_attn = th.cat(
                [values_tk, values_skip_tk.mean(dim=-1, keepdim=True)], dim=-1
            )
            self._skipped_tokens += math.prod(index.shape)

        self._total_tokens += math.prod(x.shape[0:2])

        return tokens, skip_tokens, summary_token, summary_skip_token

    def index_select(self, x, index):
        """index_select code donated by Junru.

        :x: TODO
        :index: TODO
        :returns: TODO

        """
        B, T, D = x.shape
        index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
        return th.gather(input=x, dim=1, index=index_repeat)


class GateMoE(nn.Module):
    def __init__(
        self,
        attn_blk: nn.Module,
        starting_threshold: float,
        target_threshold: float,
        is_clk_tk: bool,
        is_dist_tk: bool,
        disable: bool = False,
    ):
        super().__init__()
        self.register_buffer("_threshold", th.tensor(starting_threshold))
        self.register_buffer("threshold", th.tensor(target_threshold))

        self.comparison_fn = max if starting_threshold > target_threshold else min

        self._total_tokens = 0
        self._skipped_tokens = 0
        self.is_clk_tk = is_clk_tk
        self.is_dist_tk = is_dist_tk

        self.patch_idx = int(self.is_clk_tk) + int(self.is_dist_tk)

        self.disable = False
        self.tk_idx = None
        self.attn_blk = attn_blk

    def step(self, threshold: th.Tensor):
        self._threshold.data.copy_(threshold)

    def forward(
        self, x: th.Tensor
    ) -> typ.Tuple[th.Tensor, th.Tensor | None, th.Tensor | None, th.Tensor | None]:

        if self.disable:
            return x, None, None, None

        tokens: th.Tensor
        skip_tokens: th.Tensor | None = None
        summary_token: th.Tensor | None = None
        summary_skip_token: th.Tensor | None = None
        # cuda = x.device

        threshold = self._threshold  # if self.training else self.threshold
        density = int(x.size(1) * threshold)  # type: ignore[operator]

        logits = (
            self.attn_blk.x_cls_attn.mean(dim=1)[
                :, self.patch_idx : x.size(1) + self.patch_idx
            ]
            .detach()
            .clone()
        )

        # prob = th.sigmoid(out)

        values_tk, index = logits.topk(k=density, dim=1)
        self.tk_idx = index

        tokens = self.index_select(x, index)

        skip_tokens = None
        summary_token = None
        if x.size(1) - density > 0:
            values_skip_tk, index = logits.topk(
                k=x.size(1) - density, dim=1, largest=False
            )
            self.tk_idx = th.cat([self.tk_idx, index], dim=1)
            skip_tokens = self.index_select(x, index)
            # if x_cls is not None:
            values = values_skip_tk.softmax(dim=-1)
            summary_skip_token = (skip_tokens * values.unsqueeze(dim=-1)).sum(
                dim=1, keepdim=True
            )
            self.gate_attn = th.cat(
                [values_tk, values_skip_tk.mean(dim=-1, keepdim=True)], dim=-1
            )

            self._skipped_tokens += math.prod(index.shape)
        self._total_tokens += math.prod(x.shape[0:2])

        return tokens, skip_tokens, summary_token, summary_skip_token

    def index_select(self, x, index):
        """index_select code donated by Junru.

        :x: TODO
        :index: TODO
        :returns: TODO

        """
        B, T, D = x.shape
        index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
        return th.gather(input=x, dim=1, index=index_repeat)


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
        x = mask_and_forward(x, self.dense_gate, lambda x: self.drop_path(self.attn(x)))
        x = self.norm2(x)
        x = mask_and_forward(x, self.moe_gate, lambda x: self.drop_path(self.mlp(x)))
        return x


def forward_residule_moe(self, x):
    x = self.norm1(x)
    x = mask_and_forward(
        x,
        self.dense_gate,
        lambda x: self.drop_path(self.attn(x)),
        self.is_cls_token,
        self.is_dist_token,
    )
    x = self.norm2(x)
    x = mask_and_forward(
        x,
        self.moe_gate,
        lambda x: self.drop_path(self.mlp(x)),
        self.is_cls_token,
        self.is_dist_token,
    )
    return x


def forward_residule_moe_w_attn_loss(self, x):
    x = self.norm1(x)
    x = mask_and_forward(
        x,
        self.dense_gate,
        lambda x: self.drop_path(self.attn(x)),
        self.is_cls_token,
        self.is_dist_token,
    )

    if hasattr(self.attn, "x_cls_attn"):
        cls_attn = self.attn.x_cls_attn.mean(dim=1)  # mean over all heads
        loss = cls_attn[:, -2] - cls_attn[:, -1]  # skip_sum - non_skip_sum
        loss = loss[loss > 0].mean()
        if th.isnan(loss):
            loss = th.tensor(0).to(x.device)
        self.attn_loss = loss

    # self.attn_loss = max(
    # (self.attn.x_cls_attn[:, :, -2] - self.attn.x_cls_attn[:, :, -1]).sum(),
    # th.tensor(0).to(x.device),
    # )
    x = self.norm2(x)
    x = mask_and_forward(
        x,
        self.moe_gate,
        lambda x: self.drop_path(self.mlp(x)),
        self.is_cls_token,
        self.is_dist_token,
    )
    return x


def forward_residule_moe_w_attn_loss_v2(self, x):
    input_ = self.norm1(x)

    cls_token: th.Tensor | None = None
    dist_token: th.Tensor | None = None
    skip_tk: th.Tensor | None = None
    summary_token: th.Tensor | None = None
    summary_skip_token: th.Tensor | None = None

    patch_tk: th.Tensor
    tokens: th.Tensor

    patch_idx = 0

    if self.is_cls_token:
        cls_token = input_[:, 0].unsqueeze(dim=1)
        patch_idx += 1
    if self.is_dist_token:
        dist_token = input_[:, 1].unsqueeze(dim=1)
        patch_idx += 1

    patch_tk = input_[:, patch_idx::]

    tokens, skip_tk_attn, summary_token, summary_skip_token = self.dense_gate(
        patch_tk
    )  # , 1::])

    tokens_to_attn = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [
                    cls_token,
                    dist_token,
                    tokens,
                    summary_skip_token,
                    summary_token,
                ],
            )
        ),
        dim=1,
    )

    attended_tokens = self.drop_path(self.attn(tokens_to_attn))

    if summary_token is not None:
        attended_tokens = attended_tokens[:, 0:-1]

    if summary_skip_token is not None:
        attn_skip_token = attended_tokens[:, -1].unsqueeze(dim=1)
        attended_tokens = attended_tokens[:, 0:-1]
    else:
        attn_skip_token = None

    attended_tokens += tokens_to_attn[:, 0 : attended_tokens.size(1)]

    attended_tokens = self.norm2(attended_tokens)

    #######################
    # patch_tk = attended_tokens[:, patch_idx::]
    tokens, skip_tk_mlp, _, summary_skip_token = self.moe_gate(
        attended_tokens[:, patch_idx::]
    )  # , 1print("#2 tokens:", tokens.shape)

    tokens_to_fwd = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [
                    attended_tokens[:, 0:patch_idx],
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

    if hasattr(self.attn, "x_cls_attn"):
        cls_attn = self.attn.x_cls_attn.mean(dim=1)  # mean over all heads
        loss = cls_attn[:, -2] - cls_attn[:, -1]  # skip_sum - non_skip_sum
        loss = loss[loss > 0].mean()
        if th.isnan(loss):
            loss = th.tensor(0).to(x.device)
        self.attn_loss = loss
    return fwded_tokens


def forward_residule_moe_w_attn_loss_v3(self, x):
    input_ = self.norm1(x)

    cls_token: th.Tensor | None = None
    dist_token: th.Tensor | None = None
    skip_tk: th.Tensor | None = None
    summary_token: th.Tensor | None = None
    summary_skip_token: th.Tensor | None = None

    patch_tk: th.Tensor
    tokens: th.Tensor

    patch_idx = 0

    if self.is_cls_token:
        cls_token = input_[:, 0].unsqueeze(dim=1)
        patch_idx += 1
    if self.is_dist_token:
        dist_token = input_[:, 1].unsqueeze(dim=1)
        patch_idx += 1

    patch_tk = input_[:, patch_idx::]

    tokens, skip_tk_attn, summary_token, summary_skip_token = self.dense_gate(
        patch_tk
    )  # , 1::])

    tokens = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [cls_token, dist_token, tokens, summary_skip_token, summary_token],
            )
        ),
        dim=1,
    )
    token_attn = self.drop_path(self.attn(tokens))  # + tokens

    sum_idx = -2 if summary_token is not None else -1

    summary_skip_token_attn = token_attn[:, sum_idx].unsqueeze(dim=1)

    token_attn = (token_attn + tokens)[:, 0:sum_idx]  # get rid of sum_skip and sum_tk

    token_attn = self.norm2(token_attn)

    #######################
    patch_tk = token_attn[:, patch_idx::]
    tokens, skip_tk_mlp, _, summary_skip_token = self.moe_gate(patch_tk)  # , 1::])

    tokens = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [
                    token_attn[:, 0:patch_idx],
                    tokens,
                    summary_skip_token,
                    summary_skip_token_attn,
                ],
            )
        ),
        dim=1,
    )

    token_mlp = self.drop_path(self.mlp(tokens))

    summary_skip_token_mlp_full = token_mlp[:, -1].unsqueeze(dim=1)
    summary_skip_token_mlp_half = token_mlp[:, -2].unsqueeze(dim=1)

    token_mlp = token_mlp + tokens
    tokens = token_mlp[:, 0:-2]

    if skip_tk_attn is not None:  # and summary_skip_token is not None:
        update_skip_tk_1 = (
            skip_tk_attn
            + summary_skip_token_attn.tile(skip_tk_attn.size(1)).view(
                skip_tk_attn.shape
            )
            + summary_skip_token_mlp_full.tile(skip_tk_attn.size(1)).view(
                skip_tk_attn.shape
            )
        )
        tokens = th.cat((tokens, update_skip_tk_1), dim=1)

    if skip_tk_mlp is not None:
        update_skip_tk_2 = skip_tk_mlp + summary_skip_token_mlp_full.tile(
            skip_tk_mlp.size(1)
        ).view(skip_tk_mlp.shape)

        tokens = th.cat((tokens, update_skip_tk_2), dim=1)

    if hasattr(self.attn, "x_cls_attn"):
        cls_attn = (
            self.attn.x_cls_attn.mean(dim=1)[:, 0:-1].softmax(dim=-1).detach().clone()
        )  # mean over all heads
        self.attn_loss = (
            F.kl_div(
                self.dense_gate.gate_attn.log_softmax(dim=-1), cls_attn, log_target=True
            )
            * 0.1
        )
        # loss = cls_attn[:, -2] - cls_attn[:, -1]  # skip_sum - non_skip_sum
        # loss = loss[loss > 0].mean()
        # if th.isnan(loss):
        # loss = th.tensor(0).to(x.device)
        # self.attn_loss = loss
    return tokens


def forward_residule_vit(self, input_):
    input_ = self.norm1(input_)

    cls_token: th.Tensor | None = None
    dist_token: th.Tensor | None = None
    skip_tk: th.Tensor | None = None
    summary_token: th.Tensor | None = None
    summary_skip_token: th.Tensor | None = None

    patch_tk: th.Tensor
    tokens: th.Tensor

    patch_idx = 0

    if self.is_cls_token:
        cls_token = input_[:, 0].unsqueeze(dim=1)
        patch_idx += 1
    if self.is_dist_token:
        dist_token = input_[:, 1].unsqueeze(dim=1)
        patch_idx += 1

    patch_tk = input_[:, patch_idx::]

    tokens, skip_tk, summary_token, summary_skip_token = self.dense_gate(
        patch_tk
    )  # , 1::])

    tokens = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [cls_token, dist_token, tokens, summary_skip_token, summary_token],
            )
        ),
        dim=1,
    )
    tokens_fwd = self.drop_path(self.attn(tokens))  # + tokens

    sum_idx = -2 if summary_token is not None else -1

    attn_summary_tk = tokens_fwd[:, sum_idx].unsqueeze(dim=1)

    tokens = tokens_fwd + tokens
    tokens = self.norm2(tokens)

    tokens_fwd = self.drop_path(self.mlp(tokens))

    mlp_summary_tk = tokens_fwd[:, sum_idx].unsqueeze(dim=1)

    tokens = tokens_fwd + tokens

    num_skip_tk = 0 if skip_tk is None else skip_tk.size(1)
    tokens = tokens[:, 0 : input_.size(1) - num_skip_tk]

    if skip_tk is not None and summary_skip_token is not None:
        update_skip_tk = (
            skip_tk
            + attn_summary_tk.tile(skip_tk.size(1)).view(skip_tk.shape)
            + mlp_summary_tk.tile(skip_tk.size(1)).view(skip_tk.shape)
        )
        tokens = th.cat((tokens, update_skip_tk), dim=1)

    if hasattr(self.attn, "x_cls_attn"):
        cls_attn = self.attn.x_cls_attn.mean(dim=1)  # mean over all heads
        loss = cls_attn[:, -2] - cls_attn[:, -1]  # skip_sum - non_skip_sum
        loss = loss[loss > 0].mean()
        if th.isnan(loss):
            loss = th.tensor(0).to(x.device)
        self.attn_loss = loss
    return tokens


def mask_and_forward(
    input_: th.Tensor,
    mask_fn: typ.Callable,
    fwd_fn: typ.Callable,
    is_cls_tk: bool = False,
    is_dist_tk: bool = False,
):
    cls_token: th.Tensor | None = None
    dist_token: th.Tensor | None = None
    skip_tk: th.Tensor | None = None
    summary_token: th.Tensor | None = None
    summary_skip_token: th.Tensor | None = None

    patch_tk: th.Tensor
    tokens: th.Tensor

    patch_idx = 0

    if is_cls_tk:
        cls_token = input_[:, 0].unsqueeze(dim=1)
        patch_idx += 1
    if is_dist_tk:
        dist_token = input_[:, 1].unsqueeze(dim=1)
        patch_idx += 1

    patch_tk = input_[:, patch_idx::]

    tokens, skip_tk, summary_token, summary_skip_token = mask_fn(patch_tk)  # , 1::])

    tokens = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [cls_token, dist_token, tokens, summary_skip_token, summary_token],
            )
        ),
        dim=1,
    )

    tokens_fwd = fwd_fn(tokens)  # + tokens
    if summary_token:
        tokens_fwd = tokens_fwd[:, 0:-1]

    if skip_tk is not None and summary_skip_token is not None:
        update_skip_tk = skip_tk + tokens_fwd[:, -1].unsqueeze(dim=1).tile(
            skip_tk.size(1)
        ).view(skip_tk.shape)

        tokens = th.cat(((tokens_fwd + tokens)[:, 0:-1], update_skip_tk), dim=1)
    else:
        tokens = tokens_fwd + tokens

    return tokens


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_tiny_distilled_patch16_224


@register_model
def resmoe_small_patch16_224_expert8_attn_loss_v4_delay_start(
    pretrained=False,
    starting_threshold_dense=1.0,
    target_threshold_dense=0.9,
    starting_threshold_moe=1.0,
    target_threshold_moe=0.9,
    **kwargs,
):
    model = deit_small_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4
    drop_rate = 0.0

    moe_placement = [-1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0]

    for name, module in model.named_modules():
        if isinstance(module, Block):

            is_moe = moe_placement.pop(0)
            if is_moe == -1:
                module.mlp = CustomizedMoEMLP(
                    embed_dim,
                    embed_dim * mlp_ratio,
                    moe_num_experts=8,
                    moe_top_k=2,
                    drop=drop_rate,
                )
            else:
                module.dense_gate = Gate(
                    embed_dim,
                    1.0,
                    dropout=0.0,
                    starting_threshold=starting_threshold_dense,
                    target_threshold=target_threshold_dense,
                )

                module.moe_gate = GateMoE(
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
                bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                    module, module.__class__
                )
                setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v4_delay_start(
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

    moe_placement = [-1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 0]

    for name, module in model.named_modules():
        if isinstance(module, Block):

            is_moe = moe_placement.pop(0)
            if is_moe == -1:
                module.mlp = CustomizedMoEMLP(
                    embed_dim,
                    embed_dim * mlp_ratio,
                    moe_num_experts=8,
                    moe_top_k=2,
                    drop=drop_rate,
                )
            else:
                module.dense_gate = Gate(
                    embed_dim,
                    1.0,
                    dropout=0.0,
                    starting_threshold=starting_threshold_dense,
                    target_threshold=target_threshold_dense,
                )

                module.moe_gate = GateMoE(
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
                bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                    module, module.__class__
                )
                setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v4_keepmask(
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

    moe_placement = [1, 0] * (depth // 2)

    for name, module in model.named_modules():
        if isinstance(module, Block):
            is_moe = bool(moe_placement.pop(0))
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
                add_guass_noise=False,
                keep_prev_mask=not is_moe,
            )

            module.moe_gate = GateMoE(
                module.attn,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
                is_clk_tk=True,
                is_dist_tk=False,
                # disable=not is_moe,
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
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v4_noisy(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
                add_guass_noise=True,
            )

            module.moe_gate = GateMoE(
                module.attn,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
                is_clk_tk=True,
                is_dist_tk=False,
                disable=moe_placement[0] == 0,
            )
            if moe_placement.pop(0):

                module.mlp = CustomizedMoEMLP(
                    embed_dim,
                    embed_dim * mlp_ratio,
                    moe_num_experts=8,
                    moe_top_k=2,
                    drop=drop_rate,
                )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v4(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )

            module.moe_gate = GateMoE(
                module.attn,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
                is_clk_tk=True,
                is_dist_tk=False,
            )
            if moe_placement.pop(0):

                module.mlp = CustomizedMoEMLP(
                    embed_dim,
                    embed_dim * mlp_ratio,
                    moe_num_experts=8,
                    moe_top_k=2,
                    drop=drop_rate,
                )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v3(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
                add_guass_noise=True,
            )
            module.norm1 = nn.Identity()
            module.norm2 = nn.Identity()

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_v2(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )
            module.norm1 = nn.Identity()
            module.norm2 = nn.Identity()

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resvit_tiny_patch16_224_v2(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )

            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resvit_tiny_patch16_224_v4(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = GateMoE(
                module.attn,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
                is_clk_tk=True,
                is_dist_tk=False,
            )

            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss_v2.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_attn_loss_nonorm1(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                dropout=0.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )
            module.norm1 = nn.Identity()
            module.norm2 = nn.Identity()

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_moe_w_attn_loss.__get__(
                module, module.__class__
            )
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_distilled_patch16_224_expert8(
    pretrained=False,
    starting_threshold_dense=1.0,
    target_threshold_dense=0.9,
    starting_threshold_moe=1.0,
    target_threshold_moe=0.9,
    **kwargs,
):
    model = deit_tiny_distilled_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4
    drop_rate = 0.0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = True
            bound_method = forward_residule_moe.__get__(module, module.__class__)
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8(
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

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_moe,
                target_threshold=target_threshold_moe,
            )

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_vit.__get__(module, module.__class__)
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


@register_model
def resvit_tiny_patch16_224(
    pretrained=False, starting_threshold_dense=1.0, target_threshold_dense=0.9, **kwargs
):
    model = deit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    patch_size = 16
    embed_dim = 192
    depth = 12
    num_heads = 3
    mlp_ratio = 4
    drop_rate = 0.0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.dense_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold_dense,
                target_threshold=target_threshold_dense,
            )
            module.is_cls_token = True
            module.is_dist_token = False
            bound_method = forward_residule_vit.__get__(module, module.__class__)
            setattr(module, "forward", bound_method)
    return model
