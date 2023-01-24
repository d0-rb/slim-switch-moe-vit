import typing as typ

import numpy as np
import torch as th


def index_select(x, index):
    """index_select code donated by Junru."""
    assert len(x.shape) == 3
    B, T, D = x.shape
    index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
    return th.gather(input=x, dim=1, index=index_repeat)


def cls_attn_reordering(
    patch_tk: th.Tensor, cls_patch_attn: th.Tensor, patch_attn: th.Tensor | None = None
):
    # cls_path_attn (B x T) from B x H x N x N
    # patch_attn B x N-1 x N-1
    cls_path_attn_sorted, index = cls_patch_attn.sort(dim=1, descending=True)
    patch_tk_sorted = index_select(patch_tk, index)
    if patch_attn is not None:
        patch_attn_sorted = index_select(patch_attn, index)
        patch_attn_sorted = th.transpose(patch_attn_sorted, 1, 2)
        patch_attn_sorted = index_select(patch_attn_sorted, index)
        return patch_tk_sorted, cls_path_attn_sorted, patch_attn_sorted
    return patch_tk_sorted, cls_path_attn_sorted


def moe_flops(mlp, input_shape):
    normalized_size = np.prod(input_shape[:-1])

    gate_flops = (
        normalized_size * mlp.gate.gate.in_features * mlp.gate.gate.out_features
    )  # naive gate linear layer
    gate_flops += normalized_size * (3 * input_shape[-1] - 1)  # softmax layer

    return gate_flops


def gate_flops(gate, input_shape):
    flops = 0
    flops += np.prod(input_shape[:-1]) * gate.head.in_features * gate.head.out_features

    return flops


def resmoe_flop_hook(block, input, output):
    total_flops = 0

    total_input_scalars = np.prod(input.shape)

    norm1_flops = total_input_scalars
    if getattr(block.norm1, "affine", False) or getattr(
        block.norm1, "elementwise_affine", False
    ):
        norm1_flops *= 2

    dense_gate_flops = gate_flops(block.dense_gate, input.shape)
    # token_masking_flops = np.prod(input.shape)

    attn_mask = block.dense_gate(input)[:, :, 1]
    num_attn_tk = attn_mask.sum()
    attn_flops = (
        4 * num_attn_tk * (input.shape[2] ** 2)
        + 2 * (num_attn_tk**2) * input.shape[2]
    )
    attn_residual_flops = total_input_scalars  # flops for residual connection
    drop_path_flops = 0  # for inference, drop_path does nothing

    norm2_flops = total_input_scalars
    if getattr(block.norm2, "affine", False) or getattr(
        block.norm2, "elementwise_affine", False
    ):
        norm2_flops *= 2

    moe_gate_flops = gate_flops(block.moe_gate, input.shape)

    moe_mask = block.moe_gate(input)[:, :, 1]
    num_moe_tk = moe_mask.sum()
    mlp_flops = moe_flops(block.mlp, (num_moe_tk, *input.shape[-1]))
    moe_residual_flops = total_input_scalars  # flops for residual connection

    total_flops += norm1_flops
    total_flops += dense_gate_flops
    # total_flops += token_masking_flops
    total_flops += attn_flops
    total_flops += drop_path_flops
    total_flops += attn_residual_flops
    total_flops += norm2_flops
    total_flops += moe_gate_flops
    # total_flops += token_masking_flops
    total_flops += mlp_flops
    total_flops += drop_path_flops
    total_flops += moe_residual_flops

    return total_flops


def get_cls_token(tokens, is_attn: bool = False):
    if is_attn:
        # tokens B x T+cls x T+cls
        return tokens[:, 0, 1::], tokens[:, 1::, 1::]
    else:
        # tokens B x T + cls x Dimension
        return tokens[:, 0:1, :], tokens[:, 1::]
