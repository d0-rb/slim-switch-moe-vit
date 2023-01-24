import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as tgu
from timm.models import register_model  # type: ignore[import]
from torch_geometric.nn import GCNConv

from .selection_gates import GateImnet
from .utils import cls_attn_reordering
from .utils import get_cls_token
from .vision_transformer import Block
from .wrapper import forward_block_vanilla
from .wrapper import forward_block_w_full_attn


class GNN(nn.Module):

    """Graph convolution using Attention's attn map.
    By default this is a 2 layers GCN"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        node_density: float = 0.5,
        edge_density: float = 0.5,
        output_represenation: int = 2,
        num_layers: int = 1,
        pre_sorted: bool = True,
    ):
        """initialized GCN class.

        :dim: token's dimension
        :hidden_dim:
        :node_density: TODO
        :edge_density: TODO
        :pre_sorted: wether input and attn are sorted by on class_attn before hand

        """
        super().__init__()

        self._dim = dim
        self._hidden_dim = hidden_dim
        self._node_density = node_density
        self._edge_density = edge_density
        self._pre_sorted = pre_sorted

        self.layers_GCN = nn.ModuleList([])

        cur_dim = dim
        for i in range(num_layers - 1):
            self.layers_GCN.append(GCNConv(cur_dim, hidden_dim))
            cur_dim = hidden_dim

        self.layers_GCN.append(GCNConv(cur_dim, dim))

        self.classifier = nn.Linear(dim, output_represenation)

        self.gate = GateImnet(
            nn.Identity(),
            starting_threshold=node_density,
            target_threshold=node_density,
        )

    def feature_forward(self, nodes, edges, edge_weights):
        for gcn in self.layers_GCN[0:-1]:
            nodes = gcn(nodes, edges, edge_weights)
            nodes = F.relu(nodes)

        nodes = F.relu(self.layers_GCN[-1](nodes, edges, edge_weights))
        grouping = self.classifier(nodes)

        return grouping.softmax(dim=-1)

    def forward(self, x, attn):

        (patch_tk, skip_patch_tk), skip_patch_attn = self.get_tokens(x, attn)

        skip_patch_attn = self.edge_sparsify(skip_patch_attn)

        skip_patch_shape = skip_patch_tk.shape

        skip_patch_tk = skip_patch_tk.reshape(-1, x.size(-1))  # B * T, D
        edges, edge_weights = self.construct_graph(skip_patch_attn)

        grouping = self.feature_forward(skip_patch_tk, edges, edge_weights).view(
            skip_patch_shape[0], skip_patch_shape[1], -1
        )

        skip_token_summaries = th.transpose(grouping, -1, -2) @ skip_patch_tk.reshape(
            *skip_patch_shape
        )

        return th.cat([patch_tk, skip_token_summaries], dim=1)

    def get_tokens(self, x, attn):
        _, patch_x = get_cls_token(x)
        cls_attn, patch_attn = get_cls_token(attn, is_attn=True)

        return self.node_sparsify(patch_x, cls_attn, patch_attn)

    def node_sparsify(self, x, cls_attn, patch_attn):
        if not self._pre_sorted:
            x, cls_attn, patch_attn = cls_attn_reordering(x, cls_attn, patch_attn)

        non_skip_tokens, skip_tokens, _ = self.gate(x, cls_attn)

        n = non_skip_tokens.size(1)

        non_skip_patch_attn = patch_attn[:, 0:n, 0:n]
        skip_patch_attn = patch_attn[:, n::, n::]

        return (
            # (x, cls_attn, patch_attn),
            (non_skip_tokens, skip_tokens),
            skip_patch_attn,
        )

    def edge_sparsify(self, dense_adj):
        B, T, _ = dense_adj.shape
        if self._edge_density == 1.0:
            return dense_adj
        else:
            num_edges = int(T**2 * self._edge_density)
            dense_adj = dense_adj.reshape(B, -1)
            cutoff_threshold = dense_adj.topk(num_edges, dim=1)[0][:, -1::]
            dense_adj[dense_adj < cutoff_threshold] = 0
            return dense_adj.view(B, T, T)

    @staticmethod
    def construct_graph(batch_adj):
        shape = batch_adj.shape
        assert shape[-1] == shape[-2]
        edges, edge_attrs = tgu.dense_to_sparse(batch_adj)
        return edges, edge_attrs


def forward_gcn_drop(self, input_):
    x, attn = input_
    x = self.gcn_gate(x, attn)
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


from .model import deit_tiny_patch16_224


@register_model
def resvit_tiny_patch16_224_gcn_g3_o14(
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

    index = [3]
    output_represenation = [14]
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, Block):
            if cnt not in index:
                module.is_last = cnt == depth - 1
                bound_method = forward_block_w_full_attn.__get__(
                    module, module.__class__
                )
                module._forward = forward_block_vanilla
                setattr(module, "forward", bound_method)
            else:
                module.gcn_gate = GNN(
                    embed_dim,
                    embed_dim,
                    target_threshold_dense,
                    target_threshold_moe,
                    output_represenation=14,
                    num_layers=1,
                    pre_sorted=False,
                )
                bound_method = forward_gcn_drop.__get__(module, module.__class__)
                setattr(module, "forward", bound_method)
            cnt += 1
    return model
