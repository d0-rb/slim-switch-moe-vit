import math
import typing as typ

import torch as th
import torch.nn as nn
from fmoe import FMoETransformerMLP  # type: ignore[import]
from timm.models import register_model  # type: ignore[import]
from torch.autograd import Variable

from .model import DistilledVisionTransformer as Deit
from .vision_transformer import Block

__all__ = ["Gate", "ResBlock", "resmoe_tiny_patch16_224_expert8"]


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
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.Linear(in_dim, in_dim // 2),
            nn.Linear(in_dim, 1),
        )
        self.register_buffer("_threshold", th.tensor(starting_threshold))
        self.register_buffer("threshold", th.tensor(target_threshold))

        self._total_tokens = 0
        self._skipped_tokens = 0

        self.is_hard = is_hard
        self.disable = False

    def step(self, delta: th.Tensor):
        thresh = self._threshold - delta
        self._threshold.data.copy_(
            max(thresh, self.threshold)  # type: ignore[call-overload]
        )  # type: ignore[operator]

    def forward(
        self, x
    ) -> typ.Tuple[th.Tensor, typ.Optional[th.Tensor], typ.Optional[th.Tensor]]:
        if self.disable:
            return x, None, None

        # B, T, D = x.shape
        # cuda = x.device

        threshold = self._threshold  # if self.training else self.threshold
        density = int(x.size(1) * threshold)

        out = self.head(x)  # (B x Token x 1)
        prob = th.sigmoid(out).squeeze()

        values, index = prob.topk(k=density, dim=1)

        tokens = self.index_select(x, index)

        skip_tokens = None
        summary_token = None
        if x.size(1) - density > 0:
            values, index = prob.topk(k=x.size(1) - density, dim=1, largest=False)
            skip_tokens = self.index_select(x, index)
            values = values.softmax(dim=-1)
            summary_token = (skip_tokens * values.unsqueeze(dim=-1)).sum(
                dim=1, keepdim=True
            )

        return tokens, skip_tokens, summary_token

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
        self.is_clk_token,
        self.is_dist_token,
    )
    x = self.norm2(x)
    x = mask_and_forward(
        x,
        self.moe_gate,
        lambda x: self.drop_path(self.mlp(x)),
        self.is_clk_token,
        self.is_dist_token,
    )
    return x


def forward_residule_vit(self, x):
    x = self.norm1(x)

    def fwd_fn(x):
        x = self.drop_path(self.attn(x))
        x = self.norm2(x)
        x = self.drop_path(self.mlp(x))
        return x

    x = mask_and_forward(
        x,
        self.dense_gate,
        fwd_fn,
        self.is_clk_token,
        self.is_dist_token,
    )
    return x


def mask_and_forward(
    input_: th.Tensor,
    mask_fn: typ.Callable,
    fwd_fn: typ.Callable,
    is_cls_tk: bool = False,
    is_dist_tk: bool = False,
):
    cls_token: typ.Optional[th.Tensor] = None
    dist_token: typ.Optional[th.Tensor] = None
    skip_tk: typ.Optional[th.Tensor] = None
    summary_token: typ.Optional[th.Tensor] = None

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

    tokens, skip_tk, summary_token = mask_fn(patch_tk)  # , 1::])

    tokens = th.cat(
        list(
            filter(
                lambda x: x is not None,  # type: ignore[arg-type]
                [cls_token, dist_token, tokens, summary_token],
            )
        ),
        dim=1,
    )

    # tokens = th.cat((cls_token, tokens), dim=1)
    # if summary_token is not None:
    # tokens = th.cat((tokens, summary_token), dim=1)

    tokens_fwd = fwd_fn(tokens)  # + tokens

    if skip_tk is not None and summary_token is not None:
        update_skip_tk = skip_tk + tokens_fwd[:, -1].unsqueeze(dim=1).tile(
            skip_tk.size(1)
        ).view(skip_tk.shape)

        tokens = th.cat(((tokens_fwd + tokens)[:, 0:-1], update_skip_tk), dim=1)
    else:
        tokens = tokens_fwd + tokens

    return tokens


from .model import deit_tiny_patch16_224
from .model import deit_tiny_distilled_patch16_224


@register_model
def resmoe_tiny_distilled_patch16_224_expert8(
    pretrained=False, starting_threshold=1.0, target_threshold=0.9, **kwargs
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
                starting_threshold=starting_threshold,
                target_threshold=target_threshold,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold,
                target_threshold=target_threshold,
            )

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_clk_token = True
            module.is_dist_token = True
            bound_method = forward_residule_moe.__get__(module, module.__class__)
            setattr(module, "forward", bound_method)
    return model


@register_model
def resmoe_tiny_patch16_224_expert8(
    pretrained=False, starting_threshold=1.0, target_threshold=0.9, **kwargs
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
                starting_threshold=starting_threshold,
                target_threshold=target_threshold,
            )
            module.moe_gate = Gate(
                embed_dim,
                1.0,
                starting_threshold=starting_threshold,
                target_threshold=target_threshold,
            )

            module.mlp = CustomizedMoEMLP(
                embed_dim,
                embed_dim * mlp_ratio,
                moe_num_experts=8,
                moe_top_k=2,
                drop=drop_rate,
            )
            module.is_clk_token = True
            module.is_dist_token = False
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


@register_model
def resvit_tiny_patch16_224(
    pretrained=False, starting_threshold=1.0, target_threshold=0.9, **kwargs
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
                starting_threshold=starting_threshold,
                target_threshold=target_threshold,
            )
            module.is_clk_token = True
            module.is_dist_token = False
            bound_method = forward_residule_vit.__get__(module, module.__class__)
            setattr(module, "forward", bound_method)
    return model
