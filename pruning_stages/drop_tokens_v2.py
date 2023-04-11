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

    def construct_graph(self, adj):
        mask = (1 - th.eye(adj.shape[1])).repeat(adj.size(0), 1, 1).to(adj.device)
        adj = adj * mask
        topk_index = adj.topk(self.k, dim=-1)[1]
        binary_adj = th.scatter(th.zeros_like(adj), dim=-1, index=topk_index, value=1.0)
        binary_adj = (
            (binary_adj + binary_adj.transpose(1, 2)) > 0
        ).float()  # make symmetrical
        adj = adj * binary_adj
        edge_index, edge_attr = tgu.dense_to_sparse(adj)
        return edge_index, edge_attr

    def forward(self, embeddings, dense_adj):  # , batch):
        edge_index, edge_attr = self.construct_graph(dense_adj)
        # print(f"emb : {embeddings.shape}")
        # print(f"edge index : {edge_index.shape}")
        # print(f"batch : {batch.shape}")
        # __import__("pdb").set_trace()
        edge_index, edge_attr = self.directional_filter(
            edge_index, edge_attr, embeddings.size(0)
        )
        # print(f"edge index : {edge_index.shape}")
        # print(f"batch : {batch.shape}")
        # __import__("pdb").set_trace()

        edge_index, edge_attr = tgu.add_self_loops(
            edge_index, edge_attr, num_nodes=embeddings.size(0)
        )

        edge_index, edge_attr = tgu.sort_edge_index(edge_index, edge_attr)

        out = self.propagate(edge_index, x=embeddings, edge_weights=edge_attr)

        edge_index, edge_attr = tgu.remove_self_loops(edge_index, edge_attr)

        return out, edge_index, edge_attr

    def directional_filter(self, edge_index, edge_attrs, num_nodes):
        # edge_index, edge_attrs = tgu.to_undirected(edge_index, edge_attr=edge_attrs)

        in_degree = tgu.degree(edge_index[1, :], num_nodes)
        avg_sim = th.zeros(num_nodes).to(edge_attrs.device)
        avg_sim.index_add_(0, edge_index[1, :], edge_attrs)
        avg_sim = avg_sim / in_degree
        avg_node_sim = avg_sim[edge_index]
        # __import__("pdb").set_trace()

        cur_mask = avg_node_sim[1] > avg_node_sim[0]
        prev_mask = None

        while prev_mask is None or cur_mask.sum() != prev_mask.sum():
            edge_index = edge_index[:, cur_mask]
            edge_attrs = edge_attrs[cur_mask]
            prev_mask = cur_mask
            in_degree = tgu.degree(edge_index[1, :], num_nodes)
            avg_sim = th.zeros(num_nodes).to(edge_attrs.device)
            avg_sim.index_add_(0, edge_index[1, :], edge_attrs)
            avg_sim = avg_sim / in_degree
            avg_sim[avg_sim != avg_sim] = 0
            avg_node_sim = avg_sim[edge_index]
            # avg_node_sim = in_degree[edge_index]
            cur_mask = avg_node_sim[1] > avg_node_sim[0]

        return edge_index, edge_attrs

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

    def forward(self, tk, attention):
        batch, dim, device = tk.size(0), tk.size(-1), tk.device
        # print(f"input: {x.shape}")
        # print(f"attn: {attn.shape}")
        # import pdb; pdb.set_trace()

        # x.shape [tk+cls x Dimension]
        # attn.shape [B x tk+cls x tk] # we got rid of the first column in hook
        if self.keep_ratio == 1.0:
            return x, attn

        # tk_similarity = fast_cosine_similarity(expert_attn)
         # non_skip_tokens, skip_tokens,
            # skip_attention,
            # attention,
            # density,
        (
            cls_tk,  # 1 x D  # 1 x tk, 1 x tk
            tk, skip_tk,
            skip_tk_tk_attn,
            attention,
            density
        ) = self.get_tokens(tk, attention)
       

        batches = th.repeat_interleave(th.arange(batch), skip_tk.size(1)).to(
            device
        )

        skip_tk_shape = skip_tk.shape
        skip_tk = skip_tk.reshape(-1, dim)  # B * T, D
        # expert_patch_attn = expert_patch_attn.reshape(-1, expert_distribution.size(-1))
        skip_tk_embs, edges, edge_attr = self._aggr(
            skip_tk, skip_tk_tk_attn
        )  # batches)

        #################################################################################
        skip_tk_embs = skip_tk_embs.reshape(skip_tk_shape)
        
        num_node_grouped = int(skip_tk_shape[1] * self.grouping_ratio)
        node_degree = tgu.degree(edges[1, :], num_nodes=skip_tk.size(0))  # B * T
        # __import__("pdb").set_trace()
        node_degree = th.stack(tgu.unbatch(node_degree, batches), dim=0)  # [B x T]
        _, group_idx = node_degree.topk(
            k=num_node_grouped, dim=-1
        )

        skip_tk_embs = index_select(
            skip_tk_embs, group_idx
        )  # sorting 
        attention[:, density+1:density+1+num_node_grouped, :] = index_select(attention[:, density+1::], group_idx)
        attention = th.transpose(attention, 1, 2) 
        attention[:, density:density+num_node_grouped, :] = index_select(attention[:, density::], group_idx)
        attention = th.transpose(attention, 1, 2) 
        # truncating
        attention = attention[:, 0:density+num_node_grouped+1, 0:density+num_node_grouped] # B x trun_tk + cls x trun_tk


        return th.cat([cls_tk, tk, skip_tk_embs], dim=1), attention

    def get_tokens(self, x, attn):
        cls_tk, tokens = x[:, 0:1, :], x[:, 1::, :]
        cls_tk_attn = attn[:, 0, :]
        # cls_self_attn = attn[:, 0:1, 0:1]
        # cls_attn, patch_attn = get_cls_token(attn, is_attn=True)

        return cls_tk, *self.node_sparsify(cls_tk_attn, tokens, attn)
    
    def node_sparsify(self, sorting_vector, embeddings, attention):
    # def node_sparsify(self, x, cls_attn, patch_attn):
        # x, cls_attn, patch_attn = cls_attn_reordering(x, cls_attn, patch_attn)
        x, cls_attn, attention = cls_attn_reordering(embeddings, sorting_vector, attention)
        # x.shape [B x tk x D] (sorted)
        # cls_attn [B x tk] (sorted)
        # attention [B x tk+cls x tk] (sorted)
        
        density = int(x.size(1) * self.keep_ratio)  # type: ignore[operator]
        non_skip_tokens = x[:, 0:density]
        skip_tokens = x[:, density::]
        
        skip_attention = attention[:, density+1::, density::] 
        # [B x skip_tk x skip_tk] (plus 1 is to get rid of cls attn row)
        
        # keep_attention = attention[:, 0:density+1, 0:density+1]
        # non_skip_patch = patch_attn[:, 0:density, 0:density]
        # non_skip_cls_attn = cls_attn[:, 0:density]

        # skip_patch_attn = patch_attn[:, density::, density::]
        # skip_cls_attn = cls_attn[:, density::]

        return (
            # (non_skip_cls_attn, skip_cls_attn),
            non_skip_tokens, skip_tokens,
            skip_attention,
            attention,
            density,
            # (non_skip_patch_attn, skip_patch_attn),
        )


class DropTokens(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--attn-momentum", default=0.75, type=float)
        parser.add_argument("--finetune-epochs", default=10, type=int)
        parser.add_argument("--top-k", default=2, type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # experimental settings
        # self.layers = self.args.drop_layers
        # self.layers = [6]
        # self.layers = [2, 6, 10]
        self.layers = [3, 6, 9]
        self.keep_ratios = [[0.7,0.5, 0.5], [0.7, 0.4, 0.4], [0.7, 0.3, 0.3]]
        self.grouping_ratios = [[0.5, 0.4, 0.4], [0.5, 0.3, 0.3], [0.5, 0.2, 0.2], [0.5, 0.1, 0.1], [0.5, 0.0, 0.0]]
        # self.keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
        # self.keep_ratios = [0.5]
        # self.keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        # self.keep_ratios = [0.7, 0.6, 0.5]
        # self.grouping_ratios = [0.3]
        # self.grouping_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # 0.2, 0.1]
        # self.grouping_ratios = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        self.init(self.layers)

    def set_rate(self, keep_ratios, grouping_ratios):
        cnt = 0
        layer = self.layers
        if isinstance(keep_ratios, float):
            _keep_ratios = [keep_ratios] * len(self.layers)
        else:
            assert len(keep_ratios) == len(self.layers)
            _keep_ratios = [i for i in keep_ratios]
        if isinstance(grouping_ratios, float):
            _grouping_ratios = [grouping_ratios] * len(self.layers)
        else:
            assert len(grouping_ratios) == len(self.layers)
            _grouping_ratios = [i for i in grouping_ratios]

        for name, m in self.model.named_modules():
            if isinstance(m, (Block)):
                if cnt in layer:
                    m.drop_tokens.keep_ratio = _keep_ratios.pop(0)
                    m.drop_tokens.grouping_ratio = _grouping_ratios.pop(0)
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
        output, attn = output
    else:
        attn = module.attn.attn.detach().clone() # shape [B x Tk + cls x Tk + clk] 
        attn = attn[:, :, 1::] # shape [B x Tk+cls x Tk] : we don't care about the first column
        

    # if hasattr(module.mlp, "gate"):
    # expert_distribution = module.mlp.gate.gate_score
    # expert_distribution = expert_distribution.view(
    # x.size(0), -1, expert_distribution.size(-1)
    # )
    # token_description = th.cat([output, expert_distribution], dim=-1)
    # else:
    # token_description = output

    output, attn = module.drop_tokens(output, attn)  # token_description)
    return output, attn


def forward_attn_cumulation(self, x):
    prev_attn = 0
    if isinstance(x, (tuple)):
        x, prev_attn = x
    x = self._forward(self, x)
    attn = self.attn.attn[:, :, 1::].detach().clone() # shape B x tk+cls x tk

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
    # assert len(x.shape) == 3
    B, T, D = x.shape
    index_repeat = index.unsqueeze(-1).expand(B, index.size(1), D)
    return th.gather(input=x, dim=1, index=index_repeat)


def cls_attn_reordering(
    patch_tk: th.Tensor, cls_patch_attn: th.Tensor, patch_attn: th.Tensor | None = None
):
    # cls_path_attn (B x Tk) from B x H x N x N
    # patch_attn B x tk+cls x tk
    cls_path_attn_sorted, index = cls_patch_attn.sort(dim=1, descending=True)
    patch_tk_sorted = index_select(patch_tk, index)
    if patch_attn is not None:
        patch_attn[:, 1::, :] = index_select(patch_attn[:, 1::, :], index) 
        # patch_attn_sorted = index_select(patch_attn, index)
        patch_attn = th.transpose(patch_attn, 1, 2) # B x tk x tk+cls
        patch_attn = index_select(patch_attn, index)
        patch_attn = th.transpose(patch_attn, 1, 2) # B x tk+cls x tk
        return patch_tk_sorted, cls_path_attn_sorted, patch_attn
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
