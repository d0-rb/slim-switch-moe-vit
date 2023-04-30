import argparse
import datetime
import json
import random
import time
from functools import partial
from pathlib import Path

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import get_state_dict  # type: ignore[import]

import utils
from .base import BasePruning
from .models.vit_moe import Block
from .models.vit_moe import CustomizedGshardGate
from .models.vit_moe import CustomizedMoEMLP
from .models.vit_moe import CustomizedNaiveGate
from .benchmark import InferenceBenchmarkRunner
from .hubme import plot_and_save
from engine import evaluate
from engine import train_one_epoch


class ExpertDropping(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--expert-keep-rate",
            default=1.0,
            type=float,
            help="what percentage of experts to keep. ignored if expert-keep-count > 0",
        )
        parser.add_argument(
            "--expert-keep-count",
            default=0,
            type=int,
            help="how many experts to keep. if 0, ignore and use keep rate instead",
        )
        parser.add_argument(
            "--expert-drop-type",
            default="random",
            choices=["random", "volume", "norm", "meanshift", "cosinesim"],
            help="which expert drop type to use",
            type=str,
        )
        parser.add_argument(
            "--expert-drop-local",
            default='false',
            type=str,
            help="whether to drop locally or not",
        )
        parser.add_argument(
            "--softmax-rescale",
            default=False,
            type=bool,
            help="whether to do softmax rescaling or not",
        )

    def __init__(
        self,
        model: nn.Module,
        testloader,
        valloader,
        optimizer,
        criterion,
        loss_scaler,
        lr_scheduler,
        writer,
        args,
        model_ema,
        mixup_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model: nn.Module = model
        self.keep_rate: float | int = args.expert_keep_rate
        self.device = th.device(args.device)
        self.testLoader = testloader
        self.valLoader = valloader
        self.output_dir = Path(args.output_dir)
        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.distributed = args.distributed
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.clip_grad = args.clip_grad
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.args = args
        self.model_ema = model_ema
        self.mixup_fn = mixup_fn
        self.n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.drop_local = args.expert_drop_local.lower() == 'true'
        self.do_finetune = args.epochs > 0
        self.expert_keep_count: int = args.expert_keep_count
        self.softmax_rescale: bool = args.softmax_rescale

    def benchmark(self, loader):
        input_size = self.args.input_size
        self.args.input_size = (3, self.args.input_size, self.args.input_size)
        bench = InferenceBenchmarkRunner(
            model_name=self.args.model,
            model_object=self.model,
            data_loader=loader,
            **vars(self.args),
        )
        results = bench.run()
        self.args.input_size = input_size
        print(
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step"
        )
        return results

    def prune(self, *args, **kwargs):
        if self.expert_keep_count:
            if self.expert_keep_count < 0:
                raise ValueError("expert_keep_count, as an int, must be >= 0")
        else:
            if not 0 <= self.keep_rate <= 1:
                raise ValueError("expert_keep_rate, as a float, must be in range [0, 1]")

        dropped = self.drop(keep_rate=self.expert_keep_count if self.expert_keep_count else self.keep_rate, model=self.model)
        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedNaiveGate) or isinstance(
                module, CustomizedGshardGate
            ):
                print(
                    f"Gate {name} dropped experts: {(module.expert_mapping == -1).sum().item()}"
                )

        return dropped

    def drop(self, keep_rate: float | int, model: nn.Module):
        raise NotImplementedError("drop method must be implemented in subclass")

    # this method is no longer needed
    @staticmethod
    def correct_expert_softmax(device, self, input, output):
        top_k_idx = self.top_k_idx.reshape(*output.shape[:2], self.top_k_idx.shape[-1])
        gate_score = self.gate_score.reshape(
            *output.shape[:2], self.gate_score.shape[-1]
        )
        skipped_experts = top_k_idx == -1
        num_skipped_experts = skipped_experts.sum(
            dim=-1, keepdim=True
        )  # for each token, how many dropped experts it was sent to
        # dont rescale outputs where all or no experts were skipped
        outputs_to_rescale = (num_skipped_experts < top_k_idx.shape[-1]) & (
            num_skipped_experts > 0
        )

        # get the sum of unskipped expert softmaxes
        unskipped_expert_sum = (gate_score * ~skipped_experts).sum(dim=-1, keepdim=True)
        normalization_factor = th.ones_like(
            num_skipped_experts, device=device, dtype=output.dtype
        )
        normalization_factor[outputs_to_rescale] = (
            1 / unskipped_expert_sum[outputs_to_rescale]
        ).to(dtype=output.dtype)

        # rescale outputs for tokens where some experts were skipped
        rescaled_output = output * normalization_factor.detach()

        return rescaled_output

    @staticmethod
    def store_gate_info(self, top_k_idx, gate_score, _):
        self.top_k_idx = top_k_idx
        self.gate_score = gate_score

    def finetune(self, *args, **kwargs):
        # remove all params from optimizer except for moe
        self.optimizer.param_groups.clear()  # optim.param_group = []
        self.optimizer.state.clear()  # optim.state = defaultdict(dict)

        moe_params = []
        # add norms and cls token to optimizer, add hook to correct dropped expert softmax
        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                module.gate_hook = partial(self.store_gate_info, module)
                # module.register_forward_hook(
                # partial(self.correct_expert_softmax, self.device)
                # )
                continue
            name_hierarchy = name.split(".")
            if (
                len(name_hierarchy) == 3
                and name_hierarchy[0] == "blocks"
                and "norm" in name_hierarchy[-1]
            ):
                moe_params.extend(module.parameters())

        moe_params.append(self.model.cls_token)

        self.optimizer.add_param_group({"params": moe_params})

        print(f"Start finetuning drop-expert for {self.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(self.start_epoch, self.epochs):
            th.cuda.reset_peak_memory_stats()
            if self.distributed:
                self.valLoader.sampler.set_epoch(epoch)
                self.testLoader.sampler.set_epoch(epoch)

            val_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.valLoader,
                self.optimizer,
                self.device,
                epoch,
                self.loss_scaler,
                self.clip_grad,
                self.model_ema,
                self.mixup_fn,
                set_training_mode=False,  # eval mode
                args=self.args,
            )

            self.lr_scheduler.step(epoch)

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "checkpoint.pth"]
                print(f"Saving checkpoint to {checkpoint_paths}")
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "model_ema": get_state_dict(self.model_ema),
                            "scaler": self.loss_scaler.state_dict(),
                            "args": self.args,
                        },
                        checkpoint_path,
                    )

            test_stats = evaluate(self.testLoader, self.model, self.device)

            print(
                f"Accuracy of the network on the {len(self.testLoader.dataset)} test images: {test_stats['acc1']:.1f}%"
            )

            self.writer.log_scalar("val/loss", val_stats["loss"], epoch)
            self.writer.log_scalar("test/acc1", test_stats["acc1"], epoch)

            if "loss_attn" in val_stats:
                self.writer.log_scalar("train/loss_attn", val_stats["loss_attn"], epoch)

            if max_accuracy < test_stats["acc1"]:
                # writer.add_scalar("Accuracy/test_acc1", test_stats["acc1"], epoch)
                max_accuracy = test_stats["acc1"]
                if self.output_dir:
                    checkpoint_paths = [self.output_dir / "best_checkpoint.pth"]
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master(
                            {
                                "model": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "lr_scheduler": self.lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "model_ema": get_state_dict(self.model_ema),
                                "scaler": self.loss_scaler.state_dict(),
                                "args": self.args,
                            },
                            checkpoint_path,
                        )

            print(f"Max accuracy: {max_accuracy:.2f}%")
            self.writer.log_scalar("test/acc1/max", max_accuracy, epoch)

            log_stats = {
                **{f"val_{k}": v for k, v in val_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_parameters,
            }

            if self.output_dir and utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        if not self.do_finetune:
            throughput = self.benchmark(self.testLoader)
            test_stats = evaluate(self.testLoader, self.model, self.device)

            print(
                f"Accuracy of the network on the {len(self.testLoader.dataset)} test images: {test_stats['acc1']:.1f}%"
            )

            keep_rate_or_count = self.expert_keep_count if self.expert_keep_count else self.expert_keep_rate
            self.writer.log_scalar("test/acc1", test_stats["acc1"], keep_rate_or_count)
            # self.writer.log_scalar("samples_per_sec", test_stats["samples_per_sec"], keep_rate_or_count)
            self.writer.log_scalar("samples_per_sec", throughput['samples_per_sec'], keep_rate_or_count)
            # self.writer.log_scalar(f"batch_{self.testLoader.batch_size}_time", test_stats[f"batch_{self.testLoader.batch_size}_time"], keep_rate_or_count)
            self.writer.log_scalar(f"batch_{self.testLoader.batch_size}_time", throughput['step_time'], keep_rate_or_count)

            max_accuracy = test_stats["acc1"]

            print(f"Max accuracy: {max_accuracy:.2f}%")
            self.writer.log_scalar("test/acc1/max", max_accuracy, keep_rate_or_count)

            log_stats = {
                **{f"test_{k}": v for k, v in test_stats.items()},
                "keep_rate_or_count": keep_rate_or_count,
                "n_parameters": self.n_parameters,
            }

            if self.output_dir and utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if self.distributed:
            th.distributed.barrier()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


# randomly drop experts equally among all gates
class RandomDropping(ExpertDropping):
    def drop(self, keep_rate: float | int, model: nn.Module) -> None:
        gates = [
            module
            for module in model.modules()
            if isinstance(module, CustomizedNaiveGate)
            or isinstance(module, CustomizedGshardGate)
        ]
        self.total_experts = sum(gate.tot_expert for gate in gates)
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        if self.drop_local:
            num_drop_experts_per_gate = th.linspace(
                0, num_dropped, len(gates) + 1
            ).int()[1:]
            num_drop_experts_per_gate[1:] -= num_drop_experts_per_gate[:-1].clone()

            for i, gate in enumerate(gates):
                num_drop: int = num_drop_experts_per_gate[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                valid_expert_mask = (
                    gate.expert_mapping != -1
                ).float()  # ensure we only drop experts that have not already been dropped

                dropped_experts: th.Tensor = th.multinomial(
                    valid_expert_mask, num_drop, replacement=False
                )  # indices of experts to drop

                new_mapping: th.Tensor = gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts[]
                gate.set_expert_mapping(mapping=new_mapping)
        else:
            all_experts = []
            for gate in gates:
                # only add experts that have not already been dropped to the list of eligible experts to drop
                all_experts.extend(
                    [
                        (gate, expert_idx)
                        for expert_idx in range(gate.tot_expert)
                        if gate.expert_mapping[expert_idx] != -1
                    ]
                )

            random.shuffle(all_experts)
            new_gate_mappings = {gate: gate.expert_mapping.clone() for gate in gates}

            for gate, expert in all_experts[:num_dropped]:
                new_gate_mappings[gate][expert] = -1

            for gate, new_mapping in new_gate_mappings.items():
                gate.set_expert_mapping(mapping=new_mapping)

        print(f"dropped {num_dropped} experts out of {self.total_experts}")


# run validation on model and record volume of each expert, then drop least-visited experts
class VolumeDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_experts = 0
        self.gates = {}

        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedNaiveGate) or isinstance(
                module, CustomizedGshardGate
            ):
                module.register_buffer(
                    "expert_volume", th.zeros(module.tot_expert, device=self.device)
                )

                self.total_experts += module.tot_expert
                hook = module.register_forward_hook(
                    partial(self.record_expert_volume, self.device)
                )

                self.gates[name] = (module, hook)

                if isinstance(module, CustomizedGshardGate):
                    module.return_unpruned_topk = True

    def drop(self, keep_rate: float | int, model: nn.Module):
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_volume_modules = []
        expert_volumes = th.zeros(
            (
                0,
                self.total_experts // len(self.gates),
            ),
            device=self.device,
        )
        for name, (gate, hook) in self.gates.items():
            expert_volume_modules.extend(
                [(gate, expert) for expert in range(gate.tot_expert)]
            )
            expert_volumes = th.cat(
                (expert_volumes, gate.expert_volume.unsqueeze(dim=0)), dim=0
            )
            hook.remove()

            if isinstance(gate, CustomizedGshardGate):
                gate.return_unpruned_topk = False

        print(f"dropped expert volumes:")

        if self.drop_local:
            num_drop_experts_per_gate = th.linspace(
                0, num_dropped, len(self.gates) + 1
            ).int()[1:]
            num_drop_experts_per_gate[1:] -= num_drop_experts_per_gate[:-1].clone()

            sorted_experts = th.argsort(expert_volumes, dim=-1)

            for i, (name, (gate, hook)) in enumerate(self.gates.items()):
                num_drop: int = num_drop_experts_per_gate[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = sorted_experts[i][
                    :num_drop
                ]  # indices of experts to drop

                new_mapping: th.Tensor = gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts
                gate.set_expert_mapping(mapping=new_mapping)

                print(f"{expert_volumes[i, dropped_experts]}")
        else:
            expert_volumes = th.flatten(expert_volumes)
            # drop experts with least volume
            dropped_experts = th.argsort(expert_volumes)[:num_dropped]
            dropped_across_layers = {i: 0 for i in range(len(self.gates))}
            for _dropped_expert in dropped_experts:
                dropped_expert = _dropped_expert.item()
                gate, expert = expert_volume_modules[dropped_expert]
                gate.expert_mapping[expert] = -1
                layer = dropped_expert // gate.tot_expert
                dropped_across_layers[layer] += 1

            print(f"{expert_volumes[dropped_experts]}")
            print(f"experts dropped across layers: {dropped_across_layers}")

        print(f"dropped {num_dropped} experts out of {self.total_experts}")

    @staticmethod
    def record_expert_volume(device, self, inputs, output) -> None:
        # output[0] = [B*T, topk]
        # add 1 at indices, [B*T, topk] -> [B*T, experts]
        
        if isinstance(self, CustomizedGshardGate):
            # pruned_top_k_idx, score = output
            top_k_idx = self._topk_idx
        else:
            top_k_idx, score = output
        
        top_k_scattered = th.scatter_add(
            input=th.zeros((*top_k_idx.shape[:-1], self.tot_expert), device=device),
            dim=-1,
            index=top_k_idx,
            src=th.ones((*top_k_idx.shape[:-1], self.tot_expert), device=device),
        )

        self.expert_volume += top_k_scattered.sum(
            dim=[i for i in range(len(top_k_idx.shape) - 1)]
        )  # sum over all except last (expert) dimension, [B*T, experts] -> [experts]


# run validation on model and record mean norm of each expert output, then drop lowest-norm experts
class NormDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_experts = 0
        self.moes = {}

        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                module.register_buffer(
                    "expert_norm", th.zeros(module.num_expert, device=self.device)
                )
                module.register_buffer(
                    "num_forwards", th.zeros(module.num_expert, device=self.device)
                )

                self.total_experts += module.num_expert

                old_expert_fn = module.expert_fn

                bound_method = self.expert_fn.__get__(module, module.__class__)
                setattr(module, "expert_fn", bound_method)

                self.moes[name] = (module, old_expert_fn)

    def drop(self, keep_rate: float | int, model: nn.Module):
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_norm_modules = []
        expert_norms = th.zeros(
            (
                0,
                self.total_experts // len(self.moes),
            ),
            device=self.device,
        )
        for name, (moe, old_expert_fn) in self.moes.items():
            expert_norm_modules.extend(
                [(moe, expert) for expert in range(moe.num_expert)]
            )
            expert_norms = th.cat(
                (expert_norms, moe.expert_norm.unsqueeze(dim=0)), dim=0
            )
            setattr(moe, "expert_fn", old_expert_fn)

        print(f"dropped expert norms:")

        if self.drop_local:
            num_drop_experts_per_moe = th.linspace(
                0, num_dropped, len(self.moes) + 1
            ).int()[1:]
            num_drop_experts_per_moe[1:] -= num_drop_experts_per_moe[:-1].clone()

            sorted_experts = th.argsort(expert_norms, dim=-1)

            for i, (name, (moe, old_expert_fn)) in enumerate(self.moes.items()):
                num_drop: int = num_drop_experts_per_moe[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = sorted_experts[i][
                    :num_drop
                ]  # indices of experts to drop

                new_mapping: th.Tensor = moe.gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts
                moe.gate.set_expert_mapping(mapping=new_mapping)

                print(f"{expert_norms[i, dropped_experts]}")
        else:
            expert_norms = th.flatten(expert_norms)
            # drop experts with lowest norms
            dropped_experts = th.argsort(expert_norms)[:num_dropped]
            dropped_across_layers = {i: 0 for i in range(len(self.moes))}
            for _dropped_expert in dropped_experts:
                dropped_expert = _dropped_expert.item()
                moe, expert = expert_norm_modules[dropped_expert]
                moe.gate.expert_mapping[expert] = -1
                layer = dropped_expert // moe.gate.tot_expert
                dropped_across_layers[layer] += 1

            print(f"{expert_norms[dropped_experts]}")
            print(f"experts dropped across layers: {dropped_across_layers}")

        print(f"dropped {num_dropped} experts out of {self.total_experts}")

    @staticmethod
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            experts_out = self.experts(inp, fwd_expert_count)

            base_idx = 0
            for i in range(self.num_expert):
                batch_size = fwd_expert_count[i]
                old_num_forwards = self.num_forwards[i]
                self.num_forwards[i] += batch_size
                expert_out = experts_out[base_idx : base_idx + batch_size]

                if batch_size > 0:
                    self.expert_norm[i] = self.expert_norm[i] * (
                        old_num_forwards / self.num_forwards[i]
                    ) + expert_out.norm(dim=-1, p=2).mean() * (
                        batch_size / self.num_forwards[i]
                    )

                base_idx += batch_size

            return experts_out

        if isinstance(fwd_expert_count, th.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size

        return th.cat(outputs, dim=0)


# run validation on model and record mean expert output, then drop experts that shifted mean the least
class MeanShiftDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_experts = 0
        self.moes = {}

        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                module.register_buffer(
                    "input_mean",
                    th.zeros(
                        (module.num_expert, self.model.embed_dim), device=self.device
                    ),
                )
                module.register_buffer(
                    "expert_mean",
                    th.zeros(
                        (module.num_expert, self.model.embed_dim), device=self.device
                    ),
                )
                module.register_buffer(
                    "num_forwards", th.zeros((module.num_expert), device=self.device)
                )

                self.total_experts += module.num_expert

                old_expert_fn = module.expert_fn

                bound_method = self.expert_fn.__get__(module, module.__class__)
                setattr(module, "expert_fn", bound_method)

                self.moes[name] = (module, old_expert_fn)

    def drop(self, keep_rate: float | int, model: nn.Module):
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_mean_modules = []
        expert_mean_shifts = th.zeros(
            (
                0,
                self.total_experts // len(self.moes),
            ),
            device=self.device,
        )
        for name, (moe, old_expert_fn) in self.moes.items():
            expert_mean_modules.extend(
                [(moe, expert) for expert in range(moe.num_expert)]
            )
            expert_mean_shifts = th.cat(
                (
                    expert_mean_shifts,
                    (moe.expert_mean - moe.input_mean)
                    .norm(dim=-1, p=2)
                    .unsqueeze(dim=0),
                ),
                dim=0,
            )
            setattr(moe, "expert_fn", old_expert_fn)

        print(f"dropped expert mean shift norms:")

        if self.drop_local:
            num_drop_experts_per_moe = th.linspace(
                0, num_dropped, len(self.moes) + 1
            ).int()[1:]
            num_drop_experts_per_moe[1:] -= num_drop_experts_per_moe[:-1].clone()

            sorted_experts = th.argsort(expert_mean_shifts, dim=-1)

            for i, (name, (moe, old_expert_fn)) in enumerate(self.moes.items()):
                num_drop: int = num_drop_experts_per_moe[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = sorted_experts[i][
                    :num_drop
                ]  # indices of experts to drop

                new_mapping: th.Tensor = moe.gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts
                moe.gate.set_expert_mapping(mapping=new_mapping)

                print(f"{expert_mean_shifts[i, dropped_experts]}")
        else:
            expert_mean_shifts = th.flatten(expert_mean_shifts)
            # drop experts with lowest mean shifts
            dropped_experts = th.argsort(expert_mean_shifts)[:num_dropped]
            dropped_across_layers = {i: 0 for i in range(len(self.moes))}
            for _dropped_expert in dropped_experts:
                dropped_expert = _dropped_expert.item()
                moe, expert = expert_mean_modules[dropped_expert]
                moe.gate.expert_mapping[expert] = -1
                layer = dropped_expert // moe.gate.tot_expert
                dropped_across_layers[layer] += 1

            print(f"{expert_mean_shifts[dropped_experts]}")
            print(f"experts dropped across layers: {dropped_across_layers}")

        print(f"dropped {num_dropped} experts out of {self.total_experts}")

    @staticmethod
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            experts_out = self.experts(inp, fwd_expert_count)

            base_idx = 0
            for i in range(self.num_expert):
                batch_size = fwd_expert_count[i]
                old_num_forwards = self.num_forwards[i]
                self.num_forwards[i] += batch_size
                expert_in = inp[base_idx : base_idx + batch_size]
                expert_out = experts_out[base_idx : base_idx + batch_size]

                if batch_size > 0:
                    if old_num_forwards == 0:
                        self.expert_mean[i] = expert_out.mean(dim=0)
                        self.input_mean[i] = expert_in.mean(dim=0)
                    else:
                        self.expert_mean[i] = self.expert_mean[i] * (
                            old_num_forwards / self.num_forwards[i]
                        ) + expert_out.mean(dim=0) * (batch_size / self.num_forwards[i])
                        self.input_mean[i] = self.input_mean[i] * (
                            old_num_forwards / self.num_forwards[i]
                        ) + expert_in.mean(dim=0) * (batch_size / self.num_forwards[i])

                base_idx += batch_size

            return experts_out

        if isinstance(fwd_expert_count, th.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size

        return th.cat(outputs, dim=0)


# run validation on model and record mean cosine similarity of each expert input/output, then drop most similar experts
class CosineSimilarityDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_experts = 0
        self.moes = {}

        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                module.register_buffer(
                    "expert_similarity", th.zeros(module.num_expert, device=self.device)
                )
                module.register_buffer(
                    "num_forwards", th.zeros(module.num_expert, device=self.device)
                )

                self.total_experts += module.num_expert

                old_expert_fn = module.expert_fn

                bound_method = self.expert_fn.__get__(module, module.__class__)
                setattr(module, "expert_fn", bound_method)

                self.moes[name] = (module, old_expert_fn)

    def drop(self, keep_rate: float | int, model: nn.Module):
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_similarity_modules = []
        expert_similarities = th.zeros(
            (
                0,
                self.total_experts // len(self.moes),
            ),
            device=self.device,
        )
        for name, (moe, old_expert_fn) in self.moes.items():
            expert_similarity_modules.extend(
                [(moe, expert) for expert in range(moe.num_expert)]
            )
            expert_similarities = th.cat(
                (expert_similarities, moe.expert_similarity.unsqueeze(dim=0)), dim=0
            )
            setattr(moe, "expert_fn", old_expert_fn)
        expert_similarities[expert_similarities == 0] = float("inf")

        if self.drop_local:
            num_drop_experts_per_moe = th.linspace(
                0, num_dropped, len(self.moes) + 1
            ).int()[1:]
            num_drop_experts_per_moe[1:] -= num_drop_experts_per_moe[:-1].clone()

            sorted_experts = th.argsort(expert_similarities, descending=True, dim=-1)

            for i, (name, (moe, old_expert_fn)) in enumerate(self.moes.items()):
                num_drop: int = num_drop_experts_per_moe[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = sorted_experts[i][
                    :num_drop
                ]  # indices of experts to drop

                new_mapping: th.Tensor = moe.gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts
                moe.gate.set_expert_mapping(mapping=new_mapping)

                print(f"{expert_similarities[i, dropped_experts]}")
        else:
            expert_similarities = th.flatten(expert_similarities)
            # drop experts with highest cosine similarity
            dropped_experts = th.argsort(expert_similarities, descending=True)[
                :num_dropped
            ]
            dropped_across_layers = {i: 0 for i in range(len(self.moes))}
            for _dropped_expert in dropped_experts:
                dropped_expert = _dropped_expert.item()
                moe, expert = expert_similarity_modules[dropped_expert]
                moe.gate.expert_mapping[expert] = -1
                layer = dropped_expert // moe.gate.tot_expert
                dropped_across_layers[layer] += 1

            print(f"{expert_similarities[dropped_experts]}")
            print(f"experts dropped across layers: {dropped_across_layers}")

        print(f"dropped {num_dropped} experts out of {self.total_experts}")

    @staticmethod
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            experts_out = self.experts(inp, fwd_expert_count)

            base_idx = 0
            for i in range(self.num_expert):
                batch_size = fwd_expert_count[i]
                old_num_forwards = self.num_forwards[i]
                new_num_forwards = old_num_forwards + batch_size

                expert_in = inp[base_idx : base_idx + batch_size]
                expert_out = experts_out[base_idx : base_idx + batch_size]

                if batch_size > 0:
                    mean_cosine_simlarity = F.cosine_similarity(
                        expert_in, expert_out, dim=1
                    ).mean()
                    self.expert_similarity[i] = self.expert_similarity[i] * (
                        old_num_forwards / new_num_forwards
                    ) + mean_cosine_simlarity * (batch_size / new_num_forwards)

                # base_idx += batch_size
                self.num_forwards[i] = new_num_forwards
                base_idx += batch_size

            return experts_out

        if isinstance(fwd_expert_count, th.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size

        return th.cat(outputs, dim=0)


#############


# run validation on model and record mean class attn of each expert input, then drop experts which receive least attn input
class ClassAttnDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_experts = 0
        self.moes = {}

        for name, module in self.model.named_modules():
            if isinstance(module, Block):
                moe = module.mlp
                moe.register_buffer(
                    "expert_class_attn", th.zeros(moe.num_expert, device=self.device)
                )
                moe.register_buffer(
                    "num_forwards", th.zeros(moe.num_expert, device=self.device)
                )

                self.total_experts += moe.num_expert

                old_expert_fn = moe.expert_fn

                expert_fn = partial(self.expert_fn, module.attn)
                bound_method = expert_fn.__get__(moe, moe.__class__)
                setattr(moe, "expert_fn", bound_method)

                self.moes[name] = (moe, old_expert_fn)

    def drop(self, keep_rate: float | int, model: nn.Module):
        if isinstance(keep_rate, float):
            num_dropped = int(self.total_experts * (1 - keep_rate))
        else:
            num_dropped = self.total_experts - keep_rate

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_class_attn_modules = []
        expert_class_attns = th.zeros(
            (
                0,
                self.total_experts // len(self.moes),
            ),
            device=self.device,
        )
        for name, (moe, old_expert_fn) in self.moes.items():
            expert_class_attn_modules.extend(
                [(moe, expert) for expert in range(moe.num_expert)]
            )
            expert_class_attns = th.cat(
                (expert_class_attns, moe.expert_class_attn.unsqueeze(dim=0)), dim=0
            )
            setattr(moe, "expert_fn", old_expert_fn)

        print(f"dropped expert class attns:")

        if self.drop_local:
            num_drop_experts_per_moe = th.linspace(
                0, num_dropped, len(self.moes) + 1
            ).int()[1:]
            num_drop_experts_per_moe[1:] -= num_drop_experts_per_moe[:-1].clone()

            sorted_experts = th.argsort(expert_class_attns, dim=-1)

            for i, (name, (moe, old_expert_fn)) in enumerate(self.moes.items()):
                num_drop: int = num_drop_experts_per_moe[
                    i
                ].item()  # number of experts to drop
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = sorted_experts[i][
                    :num_drop
                ]  # indices of experts to drop

                new_mapping: th.Tensor = moe.gate.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts
                moe.gate.set_expert_mapping(mapping=new_mapping)

                print(f"{expert_class_attns[i, dropped_experts]}")
        else:
            expert_class_attns = th.flatten(expert_class_attns)
            # drop experts with lowest mean class attn
            dropped_experts = th.argsort(expert_class_attns)[:num_dropped]
            dropped_across_layers = {i: 0 for i in range(len(self.moes))}
            for _dropped_expert in dropped_experts:
                dropped_expert = _dropped_expert.item()
                moe, expert = expert_class_attn_modules[dropped_expert]
                moe.gate.expert_mapping[expert] = -1
                layer = dropped_expert // moe.gate.tot_expert
                dropped_across_layers[layer] += 1

            print(f"{expert_class_attns[dropped_experts]}")
            print(f"experts dropped across layers: {dropped_across_layers}")

        print(f"dropped {num_dropped} experts out of {self.total_experts}")

    @staticmethod
    def expert_fn(attn_block, self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            experts_out = self.experts(inp, fwd_expert_count)

            base_idx = 0
            for i in range(self.num_expert):
                batch_size = fwd_expert_count[i]
                old_num_forwards = self.num_forwards[i]
                self.num_forwards[i] += batch_size
                expert_in = inp[base_idx : base_idx + batch_size]
                expert_out = experts_out[base_idx : base_idx + batch_size]

                if batch_size > 0:
                    self.expert_class_attn[i] = self.expert_class_attn[i] * (
                        old_num_forwards / self.num_forwards[i]
                    ) + F.cosine_similarity(expert_in, expert_out, dim=1).mean() * (
                        batch_size / self.num_forwards[i]
                    )

                base_idx += batch_size

            return experts_out

        if isinstance(fwd_expert_count, th.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size

        return th.cat(outputs, dim=0)
