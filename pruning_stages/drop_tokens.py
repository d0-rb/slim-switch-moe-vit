# pylint: disable=E1101
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as tgu
from torch_geometric.nn import MessagePassing

from .base import BasePruning
from .benchmark import InferenceBenchmarkRunner
from .engine import evaluate
from .engine import train_one_epoch
from .models.vision_transformer import Block


class KthAverageAggregator(MessagePassing):
    def __init__(self, k: int, num_workers=10):
        super().__init__(aggr="sum")
        self.k = k
        self._num_workers = num_workers

    def forward(self, embeddings, criteria, batch):
        edge_index = pyg.nn.knn_graph(
            criteria,
            self.k,
            batch,
            loop=False,
            cosine=True,
            num_workers=self._num_workers,
        )
        edge_index = self.directional_filter(edge_index, embeddings.size(0))
        edge_index, _ = tgu.add_self_loops(edge_index, num_nodes=embeddings.size(0))

        edge_index = tgu.sort_edge_index(edge_index)

        out = self.propagate(edge_index, x=embeddings)
        return out, edge_index

    def directional_filter(self, edge_index, num_nodes):
        edge_index = tgu.to_undirected(edge_index)

        in_degree = tgu.degree(edge_index[1, :], num_nodes)
        edge_index_degree = in_degree[edge_index]
        cur_mask = edge_index_degree[1] > edge_index_degree[0]
        prev_mask = None

        while prev_mask is None or cur_mask.sum() != prev_mask.sum():
            edge_index = edge_index[:, cur_mask]
            prev_mask = cur_mask
            in_degree = tgu.degree(edge_index[1, :], num_nodes)
            edge_index_degree = in_degree[edge_index]
            cur_mask = edge_index_degree[1] > edge_index_degree[0]

        return edge_index

        # edge_index = edge_index.T.contiguous().tolist()
        # directional_edges = []
        # for src, tgt in edge_index:
        # if degree[src] > degree[tgt]:
        # directional_edges.append([src, tgt])
        # # degree[tgt] -= 1
        # directional_edges = th.tensor(directional_edges).T
        # # __import__("pdb").set_trace()
        # return directional_edges


class GNN(nn.Module):

    """Graph convolution using Attention's attn map.
    By default this is a 2 layers GCN"""

    def __init__(self, keep_ratio=0.5, grouping_ratio=0.5, topk=2):
        """initialized GCN class."""
        super().__init__()
        self.keep_ratio = keep_ratio
        self.grouping_ratio = grouping_ratio
        self._aggr = KthAverageAggregator(k=topk)

    def feature_forward(self, skip_patch_tk, edges, edge_weights, batches):
        mean_patch_tk = self._aggr(skip_patch_tk, edges)

    def forward(self, x, cls_attn, expert_distribution):
        if self.keep_ratio == 1.0:
            return x, cls_attn

        # tk_similarity = fast_cosine_similarity(expert_attn)

        (
            cls_tk,
            (patch_tk, skip_patch_tk),
            (nsca, sca),
            expert_patch_attn,
        ) = self.get_tokens(x, cls_attn, expert_distribution)
        batches = th.repeat_interleave(th.arange(x.size(0)), skip_patch_tk.size(1)).to(
            x.device
        )
        skip_patch_shape = skip_patch_tk.shape
        skip_patch_tk = skip_patch_tk.reshape(-1, x.size(-1))  # B * T, D
        expert_patch_attn = expert_patch_attn.reshape(-1, expert_distribution.size(-1))
        avg_patch_tk, edges = self._aggr(skip_patch_tk, expert_patch_attn, batches)
        # avg_patch_tk, edges = self._aggr(skip_patch_tk, skip_patch_tk, batches)

        #################################################################################
        avg_patch_tk = avg_patch_tk.reshape(skip_patch_shape)

        node_degree = tgu.degree(edges[0, :], num_nodes=skip_patch_tk.size(0))  # B * T
        node_degree = th.stack(tgu.unbatch(node_degree, batches), dim=0)  # [B x T]
        _, group_idx = node_degree.topk(
            k=int(skip_patch_shape[1] * self.grouping_ratio), dim=-1
        )

        # skip_patch_tk = skip_patch_tk.reshape(skip_patch_shape)  # B * T, D
        skip_token_summaries = index_select(
            avg_patch_tk, group_idx
        )  # + 0.1 * index_select(skip_patch_tk, group_idx)
        sca = index_select(sca[:, :, None], group_idx).squeeze()

        return th.cat([cls_tk, patch_tk, skip_token_summaries], dim=1), th.cat(
            [nsca, sca], dim=-1
        )

    def get_tokens(self, x, cls_attn, expert_distribution):
        cls_x, patch_x = get_cls_token(x)
        _, patch_expert_dist = get_cls_token(expert_distribution)
        # cls_attn, _ = get_cls_token(cls_attn, is_attn=True)

        return cls_x, *self.node_sparsify(patch_x, cls_attn, patch_expert_dist)

    def node_sparsify(self, x, cls_attn, patch_expert_dist):
        x, cls_attn, patch_expert_dist = cls_attn_reordering(
            x, cls_attn, patch_expert_dist
        )

        density = int(x.size(1) * self.keep_ratio)  # type: ignore[operator]
        non_skip_tokens = x[:, 0:density]
        skip_tokens = x[:, density::]

        non_skip_cls_attn = cls_attn[:, 0:density]
        skip_cls_attn = cls_attn[:, density::]

        skip_patch_attn = patch_expert_dist[:, density::]

        return (
            # (x, cls_attn, patch_attn),
            (non_skip_tokens, skip_tokens),
            (non_skip_cls_attn, skip_cls_attn),
            skip_patch_attn,
        )


class DropTokens(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--attn-momentum", default=0.5, type=float)
        parser.add_argument("--finetune-epochs", default=10, type=int)
        parser.add_argument("--top-k", default=2, type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # experimental settings
        # self.layers = self.args.drop_layers
        # self.layers = [6]
        # self.layers = [2, 6, 10]
        self.layers = [2, 6, 10]
        # self.keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
        # self.keep_ratios = [0.5]
        # self.keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        self.keep_ratios = [1.0, 0.7, 0.6, 0.5]
        # self.grouping_ratios = [0.3]
        # self.grouping_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # 0.2, 0.1]
        self.grouping_ratios = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.init(self.layers)

    def set_rate(self, keep_ratios, grouping_ratios):
        cnt = 0
        layer = self.layers
        if isinstance(keep_ratios, float):
            keep_ratios = [keep_ratios] * len(self.layers)
        else:
            assert len(keep_ratios) == len(self.layers)
        if isinstance(grouping_ratios, float):
            grouping_ratios = [grouping_ratios] * len(self.layers)
        else:
            assert len(grouping_ratios) == len(self.layers)

        for name, m in self.model.named_modules():
            if isinstance(m, (Block)):
                if cnt in layer:
                    m.drop_tokens.keep_ratio = keep_ratios.pop(0)
                    m.drop_tokens.grouping_ratio = grouping_ratios.pop(0)
                cnt += 1

    def main(self):
        # evaluate(self.testloader, self.model, self.device)
        cnt = 0
        layer = self.layers

        acc_b4, acc_af, speed = self.eval(self.keep_ratios, self.grouping_ratios)

        # acc = self.eval(keep_ratios, group_ratios)
        plot_rows(
            acc_b4,
            self.keep_ratios,
            self.grouping_ratios,
            "group_ratios",
            "accuracy",
            f"acc_b4_finetune at {self.layers}",
            f"{self.args.output_dir}/acc_b4.pdf",
        )
        plot_rows(
            acc_af,
            self.keep_ratios,
            self.grouping_ratios,
            "group_ratios",
            "accuracy",
            f"acc_af_finetune at {self.layers}",
            f"{self.args.output_dir}/acc_af.pdf",
        )

        plot_rows(
            speed,
            self.keep_ratios,
            self.grouping_ratios,
            "group_ratios",
            "ms",
            "speed at different settings",
            f"{self.args.output_dir}/speed.pdf",
        )
        th.save(
            {
                "acc_b4": acc_b4,
                "acc_af": acc_af,
                "speed": speed,
                "layers": self.layers,
                "keep_ratios": self.keep_ratios,
                "group_ratios": self.grouping_ratios,
            },
            f"{self.args.output_dir}/data.pth",
        )

    def init(self, layers):

        for name, param in self.model.named_parameters():
            if not "norm" in name:
                if "cls_token" in name:
                    pass
                else:
                    param.requires_grad = False

        cnt = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (Block)):
                module.momentum = self.args.attn_momentum
                if cnt < 11:
                    bound_method = forward_attn_cumulation.__get__(
                        module, module.__class__
                    )
                else:
                    bound_method = forward_attn_drop.__get__(module, module.__class__)

                setattr(module, "forward", bound_method)
                module._forward = vanilla_forward

                if cnt in layers:
                    module.drop_tokens = GNN(topk=self.args.top_k)
                    module.register_forward_hook(hook_token_drop)

                # if cnt % 2 == 0:
                # for param in module.mlp.parameters():
                # param.requires_grad = True

                cnt += 1

    def eval(self, keep_ratios, group_ratios):
        acc_b4 = th.zeros(len(keep_ratios), len(group_ratios))
        acc_af = th.zeros(len(keep_ratios), len(group_ratios))
        speed = th.zeros(len(keep_ratios), len(group_ratios))

        org_pth = os.path.join(self.args.output_dir, "orig.pth")
        th.save(self.model.state_dict(), org_pth)

        for i, keep_ratio in enumerate(keep_ratios):
            for j, group_ratio in enumerate(group_ratios):
                self.model.load_state_dict(th.load(org_pth))
                self.optimizer.__init__(self.model.parameters(), self.args.lr)
                print(
                    f"##################### {keep_ratio=} | {group_ratio=} ###################"
                )
                self.set_rate(keep_ratio, group_ratio)
                throughput = self.benchmark()
                results_b4 = evaluate(self.testloader, self.model, self.device)
                print("fine-tune now")
                self.finetune()
                print("fine-tunecompleted")
                results_af = evaluate(self.testloader, self.model, self.device)
                acc_b4[i, j] = results_b4["acc1"]
                acc_af[i, j] = results_af["acc1"]
                speed[i, j] = throughput["step_time"]

                if keep_ratio == 1.0:
                    break

        os.remove(org_pth)

        return acc_b4, acc_af, speed

    def benchmark(self):
        bench = InferenceBenchmarkRunner(
            model_name=self.args.model,
            model_object=self.model,
            **vars(self.args),
        )
        results = bench.run()
        print(
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step"
        )
        return results

    def finetune(self):
        best_stats = None
        for epoch in range(self.args.finetune_epochs):
            th.cuda.reset_peak_memory_stats()
            if self.args.distributed:
                self.valloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.valloader,
                self.optimizer,
                self.device,
                epoch,
                self.loss_scaler,
                self.args.clip_grad,
                None,
                self.mixup_fn,
                set_training_mode=self.args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                args=self.args,
            )
            test_stats = evaluate(self.testloader, self.model, self.device)
            if best_stats is None or best_stats["acc1"] < test_stats["acc1"]:
                th.save(
                    self.model.state_dict(),
                    os.path.join(self.args.output_dir, "temp_best_ckpt.pth"),
                )
                best_stats = test_stats
        self.model.load_state_dict(
            th.load(os.path.join(self.args.output_dir, "temp_best_ckpt.pth"))
        )
        os.remove(os.path.join(self.args.output_dir, "temp_best_ckpt.pth"))


def hook_token_drop(module, input_, output):
    x = input_[0]

    if isinstance(x, tuple):
        x = x[0]

    if isinstance(output, (tuple)):
        output, cls_attn = output
    else:
        cls_attn = module.attn.cls_attn

    if hasattr(module.mlp, "gate"):
        expert_distribution = module.mlp.gate.gate_score
        expert_distribution = expert_distribution.view(
            x.size(0), -1, expert_distribution.size(-1)
        )
        token_description = th.cat([output, expert_distribution], dim=-1)
    else:
        token_description = output

    output, cls_attn = module.drop_tokens(output, cls_attn, token_description)
    return output, cls_attn


def forward_attn_cumulation(self, x):
    prev_attn = 0
    if isinstance(x, (tuple)):
        x, prev_attn = x
    x = self._forward(self, x)
    attn = self.attn.cls_attn

    patch_attn = (1 - self.momentum) * prev_attn + self.momentum * attn
    return x, patch_attn


def forward_attn_drop(self, input_):
    x = input_
    if isinstance(input_, (tuple)):
        x = input_[0]
    return self._forward(self, x)


def vanilla_forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def plot_rows(x, keep_ratios, speed, xlabel="", ylabel="", label="", save_dir=None):
    plt.clf()
    N, M = x.shape
    for i in range(N):
        plt.plot(speed, x[i].numpy(), label=f"keep_ratio{keep_ratios[i]}")
        for j in range(M):
            plt.text(
                speed[j], x[i][j], f"{x[i][j].item():.2f}", ha="center", va="bottom"
            )  # plt.yticks(np.arange(len(yticks)), labels=yticks)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(label)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def index_select(x, index):
    """index_select code donated by Junru."""
    assert len(x.shape) == 3
    B, T, D = x.shape
    index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
    return th.gather(input=x, dim=1, index=index_repeat)


def cls_attn_reordering(
    patch_tk: th.Tensor,
    cls_patch_attn: th.Tensor,
    patch_expert_dist: th.Tensor | None = None,
):
    # cls_path_attn (B x T) from B x H x N x N
    # patch_attn B x N-1 x N-1
    cls_path_attn_sorted, index = cls_patch_attn.sort(dim=1, descending=True)
    patch_tk_sorted = index_select(patch_tk, index)
    if patch_expert_dist is not None:
        patch_expert_sorted = index_select(patch_expert_dist, index)
        return patch_tk_sorted, cls_path_attn_sorted, patch_expert_sorted

    return patch_tk_sorted, cls_path_attn_sorted


def get_cls_token(tokens, is_attn: bool = False):
    if is_attn:
        # tokens B x T+cls x T+cls
        return tokens[:, 0, 1::], tokens[:, 1::, 1::]
    else:
        # tokens B x T + cls x Dimension
        return tokens[:, 0:1, :], tokens[:, 1::]


def fast_cosine_similarity(tensor):
    # Compute the L2 norm of each embedding vector
    norm = th.norm(tensor, p=2, dim=-1, keepdim=True)

    # Normalize the embedding vectors
    tensor_normalized = tensor / norm

    # Compute the cosine similarity between all pairs of vectors in the T dimension
    similarity = th.matmul(tensor_normalized, tensor_normalized.transpose(1, 2))

    return similarity
