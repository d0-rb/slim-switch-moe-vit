# pylint: disable=E1101
# mypy: disable-error-code = attr-defined
import argparse
import copy
import typing as typ

import torch as th
from tqdm import tqdm

from .base import BasePruning
from .engine import evaluate
from .engine import train_one_epoch
from .models.vision_transformer import Block
from .models.vit_moe import CustomizedMoEMLP


class ExpertMerging(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--experts-merge-threshold", default=0.5)
        parser.add_argument("--experts-merge-step", default=1)
        # add your specific argument here #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        self.prune()
        self.finetune()

    def finetune(self, *args, **kwargs):
        pass

    def prune(self, *args, **kwargs):
        print("setting baseline loss")
        baseline_info = evaluate(self.valloader, self.model, self.device)
        experts_mapping, experts_parameters = self.get_experts_mapping()
        expert_parameters_copy = copy.deepcopy(experts_parameters)
        expert_mappings_copy = copy.deepcopy(experts_mapping)

        alphas = th.linspace(start=0, end=1, steps=self.args.experts_merge_step)
        for layer, (mapping, parameters, cp_mapping, cp_parameters) in enumerate(
            zip(
                experts_mapping,
                experts_parameters,
                expert_mappings_copy,
                expert_parameters_copy,
            )
        ):
            unique_experts = mapping.unique().tolist()
            num_experts = len(unique_experts)
            stability_mx = th.zeros(
                num_experts, num_experts, self.args.experts_merge_step
            )
            for i in tqdm(range(num_experts), total=num_experts):
                expert_1 = unique_experts[i]
                # this should be actual index position within weight matrix
                for j in tqdm(range(i + 1, num_experts), total=num_experts - i - 1):
                    expert_2 = unique_experts[j]
                    stability_mx[i, j] = self._merge_and_test_pair_of_experts(
                        parameters,
                        cp_parameters,
                        mapping,
                        cp_mapping,
                        expert_1,
                        expert_2,
                        alphas,
                        baseline_info,
                    )

            mean_stability_mx = stability_mx.mean(dim=-1)
            mean_stability_mx = (
                mean_stability_mx + mean_stability_mx.T
            )  # get full matrix
            mean_stability_mx[
                mean_stability_mx > self.args.experts_merge_threshold
            ] = float("inf")
            mean_stability_mx.fill_diagonal_(float("inf"))
            score, candidates = mean_stability_mx.topk(k=1, dim=-1, largest=False)
            expert_queue = th.argsort(score.squeeze()).flip(dims=(0,)).tolist()
            seen = th.zeros(num_experts).bool()
            while expert_queue:
                expert = expert_queue.pop(0)
                scr = score[expert]
                if scr == float("inf") or seen[expert]:
                    continue
                candidate = candidates[expert].squeeze().item()
                if seen[candidate]:
                    continue
                alpha_idx = stability_mx[expert, candidate].topk(
                    k=1, dim=-1, largest=False
                )[1]
                alpha = alphas[alpha_idx].item()

                expert_src = min(expert, candidate)
                expert_tgt = max(expert, candidate)

                self._merge_experts(
                    parameters,
                    cp_parameters,
                    mapping,
                    cp_mapping,
                    expert_tgt,
                    expert_src,
                    alpha,
                )

                seen[expert] = True
                seen[candidate] = True

    def _merge_and_test_pair_of_experts(
        self,
        parameters: typ.List[th.Tensor],
        cp_parameters: typ.List[th.Tensor],
        mapping: th.Tensor,
        cp_mapping: th.Tensor,
        expert_1: int,
        expert_2: int,
        alphas: typ.List[float],
        baseline_info: typ.Dict[str, typ.Any],
    ) -> th.Tensor:
        ret = th.zeros(len(alphas)).to(self.device)
        for i, alpha in enumerate(alphas):
            self._merge_experts(
                parameters,
                cp_parameters,
                mapping,
                cp_mapping,
                expert_1,
                expert_2,
                alpha,
            )
            mapping[mapping == expert_2] = expert_1
            info = evaluate(self.valloader, self.model, self.device, verbose=False)
            ret[i] = (info["loss"] - baseline_info["loss"]) / baseline_info["loss"]
            self._unmerge_experts(
                parameters, cp_parameters, mapping, cp_mapping, expert_1
            )
        return ret

    def _merge_experts(
        self, parameters, cp_parameters, mapping, cp_mapping, expert_1, expert_2, alpha
    ):
        for j, param in enumerate(parameters):
            param[expert_1] = (
                alpha * cp_parameters[j][expert_1]
                + (1 - alpha) * cp_parameters[j][expert_2]
            )
        mapping[mapping == expert_2] = expert_1

    def _unmerge_experts(self, parameters, cp_parameters, mapping, cp_mapping, expert):
        mapping[:] = cp_mapping
        for j, param in enumerate(parameters):
            param[expert] = cp_parameters[j][expert]

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
