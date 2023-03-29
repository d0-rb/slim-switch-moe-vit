import math
import typing as typ

import torch as th
import torch.nn as nn


class ProtoGate(nn.Module):
    def __init__(
        self,
        target_threshold: float = 0.9,
        starting_threshold: float = 1.0,
        *args,
        **kwargs,
    ):

        super().__init__()
        self.register_buffer("_threshold", th.tensor(starting_threshold))
        self.register_buffer("threshold", th.tensor(target_threshold))

        self._total_tokens = 0
        self._skipped_tokens = 0

    def step(self, threshold: th.Tensor):
        self._threshold.data.copy_(threshold)

    def forward(self, x):
        """contract function"""
        raise NotImplementedError

    def index_select(self, x, index):
        """index_select code donated by Junru.
        :x: TODO
        :index: TODO
        :returns: TODO
        """
        B, T, D = x.shape
        index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
        return th.gather(input=x, dim=1, index=index_repeat)


class Gate(ProtoGate):
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

        super().__init__(
            target_threshold=target_threshold, starting_threshold=starting_threshold
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(in_dim, 1) if not keep_prev_mask else nn.Identity()
        self.register_buffer("tau", th.tensor(tau))

        self.is_hard = is_hard
        self.disable = False
        self.tk_idx = None
        self.add_guass_noise = add_guass_noise
        self.keep_prev_mask = keep_prev_mask

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


class GateMoE(ProtoGate):
    def __init__(
        self,
        attn_blk: nn.Module,
        starting_threshold: float,
        target_threshold: float,
        is_clk_tk: bool,
        is_dist_tk: bool,
        disable: bool = False,
    ):
        super().__init__(
            target_threshold=target_threshold, starting_threshold=starting_threshold
        )

        self.is_clk_tk = is_clk_tk
        self.is_dist_tk = is_dist_tk

        self.patch_idx = int(self.is_clk_tk) + int(self.is_dist_tk)

        self.disable = False
        self.tk_idx = None
        self.attn_blk = attn_blk

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


class GateImnet(ProtoGate):
    def __init__(
        self,
        norm_layer: typ.Callable,
        num_groups: int = 1,
        disable: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm = norm_layer
        self.num_groups = num_groups
        self.group_size = -1
        self.disable = disable

    def forward(self, x, class_attn):
        if self.disable:
            return x, None, None
        self._total_tokens += math.prod(x.shape[0:2])
        # assume x and attn are sorted by descending
        threshold = self._threshold  # if self.training else self.threshold
        density = int(x.size(1) * threshold)  # type: ignore[operator]

        tokens = x[:, 0:density]
        self.tk_idx = th.arange(x.size(1))
        self.gate_attn = class_attn

        skip_tokens = None
        summary_token = None
        if (num_skipped := (x.size(1) - density)) > 0:

            skip_tokens = x[:, density::]
            values = class_attn[:, density::].clone()
            values = values.unsqueeze(dim=-1)
            self._skipped_tokens += math.prod(values.shape)
            if num_skipped <= self.num_groups or self.num_groups == 1:
                summary_skip_token = self.norm(
                    (skip_tokens * values).sum(dim=1, keepdim=True)
                )
            else:
                self.group_size = num_skipped // self.num_groups
                values = th.split(values, self.group_size, dim=1)
                group_tokens = th.split(skip_tokens, self.group_size, dim=1)
                #
                even_v, odd_v = th.stack(values[0:-1], dim=0), values[-1]
                even_t, odd_t = th.stack(group_tokens[0:-1], dim=0), group_tokens[-1]

                even_v_t = (even_t * even_v).sum(dim=2).transpose(0, 1)
                # ^ creates (B, num_group-1, D)
                odd_v_t = (odd_t * odd_v).sum(dim=1, keepdim=True)
                # ^ creates (B, 1, D)
                summary_skip_token = self.norm(th.cat([even_v_t, odd_v_t], dim=1))
                # ^ creates normalized (B, num_group, D)

        else:
            summary_skip_token = None
            self._skipped_tokens = 0.0

        return tokens, skip_tokens, summary_skip_token
