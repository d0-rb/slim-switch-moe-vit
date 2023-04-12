# pylint: disable=E1101
import argparse
import copy
import typing as typ

import torch as th

from .base import BasePruning
from .engine import evaluate
from .engine import train_one_epoch
from .models.vision_transformer import Block
from .models.vit_moe import CustomizedMoEMLP


class ExpertMerging(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--experts-merge-threshold", default=0.5)
        parser.add_argument("--experts-merge-step", default=5)
        # add your specific argument here #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        self.prune()
        self.finetune()

    def finetune(self, *args, **kwargs):
        pass

        # x = self.htoh4(inp, fwd_expert_count)
        # x = self.activation(x)
        # x = self.h4toh(x, fwd_expert_count)

    def prune(self, *args, **kwargs):
        print("setting baseline loss")
        baseline_info = evaluate(self.valloader, self.model, self.device)
        experts_mapping, experts_parameters = self.get_experts_mapping()
        expert_parameters_copy = copy.deepcopy(experts_parameters)
        expert_mappings_copy = copy.deepcopy(experts_mapping)

        alpha = th.arange(start=0, end=1, step=self.args.experts_merge_step)
        for layer, (mapping, parameters) in enumerate(
            zip(experts_mapping, experts_parameters)
        ):
            unique_experts = mapping.unique().tolist()
            num_experts = len(unique_experts)
            stability_mx = th.zeros(
                num_experts, num_experts, self.args.experts_merge_step
            )
            for i in range(num_experts):
                e1 = unique_experts[i]
                # this should be actual index position within weight matrix
                for j in range(i + 1, num_experts):
                    e2 = unique_experts[j]
                    for ii, a in enumerate(alpha):
                        for pi, p in enumerate(parameters):
                            p[e1] = (
                                a * expert_parameters_copy[layer][pi][e1]
                                + (1 - a) * expert_parameters_copy[layer][pi][e2]
                            )

                        __import__("pdb").set_trace()
                        # TODO: check if we are writing to a reference or to a copy
                        mapping[mapping == e2] = e1
                        info = evaluate(self.valloader, self.model, self.device)
                        stability_mx[i, j, ii] = (
                            info["loss"] - baseline_info["loss"]
                        ) / baseline_info["loss"]
                        mapping[:] = expert_mappings_copy[layer]

                        for pi, p in enumerate(parameters):
                            p[e1] = expert_parameters_copy[layer][pi][e1]

            stability_mx = stability_mx.mean(dim=-1)
            stability_mx = stability_mx + stability_mx.T  # get full matrix
            stability_mx[stability_mx > self.args.experts_merge_threshold] = float(
                "inf"
            )
            stability_mx.diag()[:] = float("inf")
            score, idx = stability_mx.topk(k=1, dim=-1, largest=False)
            __import__("pdb").set_trace()

    @staticmethod
    def weight_interpolation(
        w1: typ.Dict[str, th.Tensor], w2: typ.Dict[str, th.Tensor], alpha: int
    ) -> typ.Dict:
        ret = {}
        for k in w1:
            ret[k] = alpha * w1[k] + (1 - alpha) * w2[k]
        return ret

    def get_experts_mapping(self):
        expert_mappings = []
        experts_weights = []

        for name, module in self.model.named_modules():
            if isinstance(module, Block) and isinstance(module.mlp, CustomizedMoEMLP):
                expert_mappings.append(module.mlp.gate.expert_mapping)
                experts_weights.append(list(module.mlp.experts.parameters()))

        expert_mappings = expert_mappings[::-1]
        experts_weights = experts_weights[::-1]
        return expert_mappings, experts_weights
