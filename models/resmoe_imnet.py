from __future__ import annotations

import typing as typ

import torch as th
from timm.models import register_model  # type: ignore[import]

from .resMoE import CustomizedMoEMLP
from .selection_gates import GateImnet
from .utils import cls_attn_reordering
from .utils import get_cls_token
from .vision_transformer import Block
from .wrapper import *


def forward_residule_moe(
    self, input_: typ.Tuple[th.Tensor, th.Tensor]
):  # x, class_attn):

    x, class_attn = input_
    x = self.norm1(x)

    cls_token, patch_tk = get_cls_token(x)
    patch_tk, class_attn = cls_attn_reordering(patch_tk, class_attn)

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

    return fwded_tokens, class_attn


def forward_residule_moe_multi_group(
    self, input_: typ.Tuple[th.Tensor, th.Tensor]
):  # x, class_attn):

    x, class_attn = input_
    x = self.norm1(x)

    cls_token, patch_tk = x[:, 0], x[:, 1::]
    cls_token = cls_token.unsqueeze(dim=1)
    patch_tk, class_attn = cls_attn_reordering(patch_tk, class_attn)

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

    match (summary_skip_token is not None):
        case True:
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
        case default:
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
    group_size = self.moe_gate.group_size

    match (attn_skip_token is None):
        case True:
            summary_skip_token = None
        case default:
            summary_skip_token_mlp_full = fwded_tokens[:, -1].unsqueeze(dim=1)
            fwded_tokens = fwded_tokens[:, 0:-1]

    match (summary_skip_token is None):
        case True:
            summary_skip_token_mlp_half = None
            num_group = 1
        case default:
            num_group = summary_skip_token.size(1)
            summary_skip_token_mlp_half = fwded_tokens[:, -num_group::]
            fwded_tokens = fwded_tokens[:, 0:-num_group]

    fwded_tokens += tokens_to_fwd[:, 0 : fwded_tokens.size(1)]
    ret: typ.List[th.Tensor] = [fwded_tokens]

    if skip_tk_mlp is not None and summary_skip_token_mlp_half is not None:
        match (num_group):
            case 1:
                update_skip_tk_2 = skip_tk_mlp + summary_skip_token_mlp_half.tile(
                    skip_tk_mlp.size(1)
                ).view(skip_tk_mlp.shape)
                ret.append(update_skip_tk_2)

                # fwded_tokens = th.cat((fwded_tokens, update_skip_tk_2), dim=1)
            case _:
                # group_sum = th.split(summary_skip_token_mlp_half, num_group, dim=1)
                group_sum = summary_skip_token_mlp_half.transpose(0, 1)
                group_skips = th.split(skip_tk_mlp, group_size, dim=1)
                even_su, odd_su = (
                    group_sum[0:-1].unsqueeze(dim=2),  # G-1, B, 1, D
                    group_sum[-1].unsqueeze(1),  # B,1,D
                )
                even_sk, odd_sk = (
                    th.stack(group_skips[0:-1], dim=0),  # G-1, B, T, D
                    group_skips[-1],  # B, T, D
                )
                even_update = (
                    even_sk + even_su.tile(even_sk.size(2)).view(even_sk.shape)
                ).reshape(x.size(0), -1, x.size(-1))
                # ^ B, T * G-1, D
                odd_update = odd_sk + odd_su.tile(odd_sk.size(1)).view(odd_sk.shape)
                ret.extend((even_update, odd_update))

    if skip_tk_attn is not None:  # and summary_skip_token is not None:
        update_skip_tk_1 = (
            skip_tk_attn
            + attn_skip_token.tile(skip_tk_attn.size(1)).view(skip_tk_attn.shape)
            + summary_skip_token_mlp_full.tile(skip_tk_attn.size(1)).view(
                skip_tk_attn.shape
            )
        )

        ret.append(update_skip_tk_1)

        # fwded_tokens = th.cat((fwded_tokens, update_skip_tk_1), dim=1)
    ret = th.cat(ret, dim=1)  # type: ignore[assignment]

    class_attn = th.cat(
        (attended_patch_attention, class_attn[:, attended_patch_attention.size(1) : :]),
        dim=1,
    )

    return ret, class_attn


def mask_and_forward(
    cls_token: th.Tensor,
    patch_tk: th.Tensor,
    class_attn: th.Tensor,
    gate: typ.Callable,
    dense_fn: typ.Callable,
    extra_tokens: typ.List[th.Tensor] | None = None,
):
    if gate is not None:
        tokens, skip_tokens, summary_token = gate(patch_tk, class_attn)  # , 1::])
    else:
        tokens = patch_tk
        skip_tokens = summary_token = None

    if extra_tokens is None:
        extra_tokens = []

    tokens_to_attn = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [cls_token, tokens, summary_token, *extra_tokens],
            )
        ),
        dim=1,
    )

    process_tokens = dense_fn(tokens_to_attn)

    return process_tokens, tokens_to_attn, skip_tokens, summary_token


def flexible_group_addition(
    token: th.Tensor,
    tokens_to_add: typ.List[th.Tensor | typ.Any] | None = None,
    num_group: int = 1,
    group_size: int = 0,
):
    if token is None:
        return []
    tokens_to_add = list(filter(lambda x: x is not None, tokens_to_add))

    if len(token.shape) == 3:
        B, T, D = token.size()
    else:
        B, T = token.size()

    if num_group > 1:
        group_skips = th.split(token, group_size, dim=1)
        even_sk, odd_sk = (
            th.stack(group_skips[0:-1], dim=0),  # G-1, B, T, D
            group_skips[-1],  # B, T, D
        )
    else:
        even_sk = token
        odd_sk = None

    for tk2add in tokens_to_add:
        if num_group > 1:
            group_sum = tk2add.transpose(0, 1)
            even_su, odd_su = (
                group_sum[0:-1].unsqueeze(dim=2),  # G-1, B, 1, D
                group_sum[-1].unsqueeze(1),  # B,1,D
            )
            even_sk = even_sk + even_su.tile(even_sk.size(2)).view(even_sk.shape)
            # ^ B, T * G-1, D
            odd_sk = odd_sk + odd_su.tile(odd_sk.size(1)).view(odd_sk.shape)
        else:
            even_sk = even_sk + tk2add.tile(T).view(token.shape)

    if num_group == 1:
        return [even_sk]
    else:
        if len(token.shape) == 2:
            even_sk = even_sk.reshape(B, -1)
        else:
            even_sk = even_sk.reshape(B, -1, D)
        return [even_sk, odd_sk]


def update_attention(class_attn, cur_attn, is_skipped, num_group, group_size):

    if is_skipped:  # dense_summary_tk is not None:

        _token_attn = cur_attn[:, 0:-num_group]
        token_attn = class_attn[:, 0 : _token_attn.size(1)]
        token_attn = token_attn + _token_attn

        _skip_attn = cur_attn[:, -num_group::]
        skip_attn = class_attn[:, token_attn.size(1) : :]

        skip_attn = flexible_group_addition(
            skip_attn, [_skip_attn], num_group, group_size
        )

        class_attn = th.cat((token_attn, *skip_attn), dim=1) * 0.5
    else:

        class_attn = (class_attn + cur_attn) / 2.0

    return class_attn


def forward_spatial_grouping(
    self, input_: typ.Tuple[th.Tensor, th.Tensor]
):  # x, class_attn):

    x, class_attn = input_
    x = self.norm1(x)
    cls_token, patch_tk = get_cls_token(x)
    patch_tk, class_attn = cls_attn_reordering(patch_tk, class_attn)

    attended_tokens, tokens_to_attn, dense_skip_tk, dense_summary_tk = mask_and_forward(
        cls_token,
        patch_tk,
        class_attn,
        self.dense_gate,
        lambda x: self.drop_path(self.attn(x)),
    )

    dense_group_size = self.dense_gate.group_size
    dense_num_group = 0 if dense_summary_tk is None else dense_summary_tk.size(1)

    cur_attn = self.attn.x_cls_attn.sum(dim=1)[
        :, 1::
    ]  # B x patch_tk_size() + othershit

    class_attn = update_attention(
        class_attn,
        cur_attn,
        dense_summary_tk is not None,
        dense_num_group,
        dense_group_size,
    )
    if dense_summary_tk is not None:
        attn_skip_token = attended_tokens[:, -dense_num_group::, :]
        attended_tokens = attended_tokens[:, 0:-dense_num_group, :]
    else:
        attn_skip_token = None

    attended_tokens = attended_tokens + tokens_to_attn[:, 0 : attended_tokens.size(1)]
    attended_tokens = self.norm2(attended_tokens)

    #######################
    attended_cls_token, attended_patch_tk = get_cls_token(attended_tokens)
    attended_patch_tk, attended_patch_attention = cls_attn_reordering(
        attended_patch_tk, class_attn[:, 0 : attended_patch_tk.size(1)]
    )
    # patch_tk = attended_tokens[:, patch_idx::]
    ffn_tokens, ffn_residual, ffn_skip_tk, ffn_summary_token = mask_and_forward(
        attended_cls_token,
        attended_patch_tk,
        attended_patch_attention,
        self.moe_gate,
        lambda x: self.drop_path(self.mlp(x)),
        [attn_skip_token],
    )

    moe_group_size = self.moe_gate.group_size

    if dense_summary_tk is None:
        summary_skip_token_mlp_full = None
    else:
        summary_skip_token_mlp_full = ffn_tokens[:, -dense_num_group::]
        ffn_tokens = ffn_tokens[:, 0:-dense_num_group]

    if ffn_summary_token is None:
        summary_skip_token_mlp_half = None
        moe_num_groups = 1
    else:
        moe_num_groups = ffn_summary_token.size(1)
        summary_skip_token_mlp_half = ffn_tokens[:, -moe_num_groups::]
        ffn_tokens = ffn_tokens[:, 0:-moe_num_groups]

    ffn_tokens += ffn_residual[:, 0 : ffn_tokens.size(1)]
    ret: typ.List[th.Tensor] = [ffn_tokens]
    ret.extend(
        flexible_group_addition(
            ffn_skip_tk, [summary_skip_token_mlp_half], moe_num_groups, moe_group_size
        )
    )
    ret.extend(
        flexible_group_addition(
            dense_skip_tk,
            [attn_skip_token, summary_skip_token_mlp_full],
            dense_num_group,
            dense_group_size,
        )
    )

    # fwded_tokens = th.cat((fwded_tokens, update_skip_tk_1), dim=1)
    ret = th.cat(ret, dim=1)  # type: ignore[assignment]

    class_attn = th.cat(
        (attended_patch_attention, class_attn[:, attended_patch_attention.size(1) : :]),
        dim=1,
    )

    return ret, class_attn


from .model import deit_tiny_patch16_224
from .model import deit_small_patch16_224
from .model import deit_base_patch16_224
from .model import deit_tiny_distilled_patch16_224


@register_model
def resmoe_tiny_patch16_224_expert8_imnet_multi5_v4(
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
    # receptive_field = [*([[7, 7]] * 4), *([[49, 7]] * 4), *([[98, 7]] * 4)]
    # receptive_field = [*([[2, 1]] * 4), *([[2, 3]] * 4), *([[2, 3]] * 4)]
    receptive_field = [*([[1, 1]] * 4), *([[1, 1]] * 4), *([[1, 1]] * 4)]
    index = 0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            is_moe = moe_placement.pop(0)
            receptive_size = receptive_field.pop(0)
            print(receptive_size)
            if index < 4:
                print(f"{name=}")
                bound_method = forward_block_w_attn.__get__(module, module.__class__)
                module._forward = forward_block_vanilla
                setattr(module, "forward", bound_method)
            else:
                print(f"dense-gate is disabled: {index<4}")
                module.dense_gate = GateImnet(
                    module.norm1,
                    starting_threshold=starting_threshold_dense,
                    target_threshold=target_threshold_dense,
                    num_groups=receptive_size[0],
                    disable=index < 4,
                )

                module.moe_gate = GateImnet(
                    module.norm2,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=receptive_size[1],
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
                    bound_method = forward_spatial_grouping.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                else:
                    print(f"{name=} type 3")
                    module._forward = forward_spatial_grouping
                    bound_method = forward_block_no_attn.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
            index += 1
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_imnet_multi4_v4(
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
    # receptive_field = [*([[7, 7]] * 4), *([[49, 7]] * 4), *([[98, 7]] * 4)]
    receptive_field = [*([[2, 1]] * 4), *([[2, 3]] * 4), *([[2, 3]] * 4)]
    # receptive_field = [*([[1, 1]] * 4), *([[1, 1]] * 4), *([[1, 1]] * 4)]
    index = 0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            is_moe = moe_placement.pop(0)
            receptive_size = receptive_field.pop(0)
            print(receptive_size)
            if index < 4:
                print(f"{name=}")
                bound_method = forward_block_w_attn.__get__(module, module.__class__)
                module._forward = forward_block_vanilla
                setattr(module, "forward", bound_method)

            else:
                print(f"dense-gate is disabled: {index<4}")
                module.dense_gate = GateImnet(
                    module.norm1,
                    starting_threshold=starting_threshold_dense,
                    target_threshold=target_threshold_dense,
                    num_groups=receptive_size[0],
                    disable=index < 4,
                )

                module.moe_gate = GateImnet(
                    module.norm2,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=receptive_size[1],
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
                    bound_method = forward_spatial_grouping.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                else:
                    print(f"{name=} type 3")
                    module._forward = forward_spatial_grouping
                    bound_method = forward_block_no_attn.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
            index += 1
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_imnet_multi2_v4(
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
    # receptive_field = [*([[7, 7]] * 4), *([[49, 7]] * 4), *([[98, 7]] * 4)]
    receptive_field = [*([[2, 3]] * 4), *([[2, 3]] * 4), *([[2, 3]] * 4)]
    index = 0

    for name, module in model.named_modules():
        if isinstance(module, Block):
            is_moe = moe_placement.pop(0)
            receptive_size = receptive_field.pop(0)
            print(receptive_size)
            if index < 4:
                print(f"{name=}")
                bound_method = forward_block_w_attn.__get__(module, module.__class__)
                module._forward = forward_block_vanilla
                setattr(module, "forward", bound_method)

            else:
                module.dense_gate = GateImnet(
                    module.norm1,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=receptive_size[0],
                )

                module.moe_gate = GateImnet(
                    module.norm2,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=receptive_size[1],
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
                    bound_method = forward_spatial_grouping.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                else:
                    print(f"{name=} type 3")
                    module._forward = forward_spatial_grouping
                    bound_method = forward_block_no_attn.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
            index += 1
    return model


@register_model
def resmoe_tiny_patch16_224_expert8_imnet_multi_v4(
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
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=1,
                )

                module.moe_gate = GateImnet(
                    module.norm2,
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                    num_groups=3,
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
                    bound_method = forward_residule_moe_multi_group.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                else:
                    print(f"{name=} type 3")
                    module._forward = forward_residule_moe_multi_group
                    bound_method = forward_block_no_attn.__get__(
                        module, module.__class__
                    )
                    setattr(module, "forward", bound_method)
                index += 1
    return model


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
                    starting_threshold=starting_threshold_moe,
                    target_threshold=target_threshold_moe,
                )

                module.moe_gate = GateImnet(
                    module.norm2,
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
