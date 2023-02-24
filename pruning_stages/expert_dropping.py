from functools import partial
import argparse
import json
import torch as th
import torch.nn as nn
import time
import datetime
from timm.utils import get_state_dict  # type: ignore[import]
from pathlib import Path

import utils
from engine import evaluate, train_one_epoch

from .base import BasePruning
from models.vit_moe import CustomizedNaiveGate, CustomizedGshardGate, CustomizedMoEMLP


class ExpertDropping(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--expert-keep-rate", default=1., type=float, help='what percentage of experts to keep')

    def __init__(self, model: nn.Module, testloader, valloader, optimizer, criterion, loss_scaler, lr_scheduler, writer, args, model_ema, mixup_fn, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = model
        self.keep_rate = args.expert_keep_rate
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
        self.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.do_finetune = args.epochs > 0

    def prune(self, *args, **kwargs):
        if not 0 <= self.keep_rate <= 1:
            raise ValueError('expert_keep_rate must be in range [0, 1]')
        
        return self.drop(keep_rate=self.keep_rate, model=self.model)

    def drop(self, keep_rate: float | int, model: nn.Module):
        raise NotImplementedError('drop method must be implemented in subclass')
    
    def finetune(self, *args, **kwargs):
        if not self.do_finetune:
            return
        
        # remove all params from optimizer except for moe
        self.optimizer.param_groups.clear() # optim.param_group = []
        self.optimizer.state.clear() # optim.state = defaultdict(dict)

        moe_params = []
        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                moe_params.extend(module.parameters())
                continue
            name_hierarchy = name.split('.')
            if len(name_hierarchy) == 3 and name_hierarchy[0] == 'block' and 'norm' in name_hierarchy[-1]:
                moe_params.extend(module.parameters())

        self.optimizer.add_param_group(
            {'params' : moe_params}
        )

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

        if self.distributed:
            th.distributed.barrier()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        

# randomly drop experts equally among all gates
class RandomDropping(ExpertDropping):
    def drop(self, keep_rate: float | int, model: nn.Module) -> None:
        num_dropped = 0
        total_experts = 0
        for name, module in model.named_modules():
            if isinstance(module, CustomizedNaiveGate) or isinstance(module, CustomizedGshardGate):
                num_drop: int = int(module.tot_expert * (1 - keep_rate))  # number of experts to drop
                num_dropped += num_drop
                total_experts += module.tot_expert
                if num_drop == 0:
                    continue

                dropped_experts: th.Tensor = th.multinomial(th.ones(module.tot_expert), num_drop, replacement=False)  # indices of experts to drop

                new_mapping: th.Tensor = module.expert_mapping.clone()
                new_mapping[dropped_experts] = -1  # drop experts[]
                module.set_expert_mapping(mapping=new_mapping)
        
        print(f'dropped {num_dropped} experts out of {total_experts}')


# run validation on model and record volume of each expert, then drop least-visited experts
class VolumeDropping(ExpertDropping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_experts = 0
        self.gates = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedNaiveGate) or isinstance(module, CustomizedGshardGate):
                module.register_buffer('expert_volume', th.zeros(module.tot_expert, device=self.device))
                
                self.total_experts += module.tot_expert
                hook = module.register_forward_hook(partial(self.record_expert_volume, self.device))

                self.gates[name] = (module, hook)

    def drop(self, keep_rate: float | int, model: nn.Module):
        num_dropped = int(self.total_experts * (1 - keep_rate))

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_volume_modules = []
        expert_volumes = th.zeros((0,), device=self.device)
        for name, (gate, hook) in self.gates.items():
            expert_volume_modules.extend([(gate, expert) for expert in range(gate.tot_expert)])
            expert_volumes = th.cat((expert_volumes, gate.expert_volume), dim=0)
            hook.remove()
        
        # drop experts with least volume
        dropped_experts = th.argsort(expert_volumes)[:num_dropped]
        for dropped_expert in dropped_experts:
            gate, expert = expert_volume_modules[dropped_expert]
            gate.expert_mapping[expert] = -1
        
        print(f'dropped {num_dropped} experts out of {self.total_experts}')
        print(f'dropped expert volumes: {expert_volumes[dropped_experts]}')
    
    @staticmethod
    def record_expert_volume(device, self, inputs, output) -> None:
        # output[0] = [B*T, topk]
        # add 1 at indices, [B*T, topk] -> [B*T, experts]
        top_k_idx, score = output
        top_k_scattered = th.scatter_add(input=th.zeros((*top_k_idx.shape[:-1], self.tot_expert), device=device), dim=-1, index=top_k_idx, src=th.ones((*top_k_idx.shape[:-1], self.tot_expert), device=device))

        self.expert_volume += top_k_scattered.sum(dim=[i for i in range(len(top_k_idx.shape) - 1)])  # sum over all except last (expert) dimension, [B*T, experts] -> [experts]



# run validation on model and record mean norm of each expert output, then drop lowest-norm experts
class NormDropping(ExpertDropping):
    def __init__(self, *args, valloader, **kwargs):
        super().__init__(*args, **kwargs)

        self.valLoader = valloader
        self.total_experts = 0
        self.moes = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, CustomizedMoEMLP):
                module.register_buffer('expert_norm', th.zeros(module.num_expert, device=self.device))
                module.register_buffer('num_forwards', th.zeros(module.num_expert, device=self.device))
                
                self.total_experts += module.num_expert

                old_expert_fn = module.expert_fn
                
                bound_method = self.expert_fn.__get__(
                    module, module.__class__
                )
                setattr(module, "expert_fn", bound_method)

                self.moes[name] = (module, old_expert_fn)

    def drop(self, keep_rate: float | int, model: nn.Module):
        num_dropped = int(self.total_experts * (1 - keep_rate))

        self.model.eval()
        for samples, targets in self.valLoader:
            samples = samples.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)

            with th.no_grad():
                outputs = model(samples)

        expert_norm_modules = []
        expert_norms = th.zeros((0,), device=self.device)
        for name, (moe, old_expert_fn) in self.moes.items():
            expert_norm_modules.extend([(moe, expert) for expert in range(moe.num_expert)])
            expert_norms = th.cat((expert_norms, moe.expert_norm), dim=0)
            setattr(moe, "expert_fn", old_expert_fn)
        
        # drop experts with lowest norms
        dropped_experts = th.argsort(expert_norms)[:num_dropped]
        for dropped_expert in dropped_experts:
            moe, expert = expert_norm_modules[dropped_expert]
            moe.gate.expert_mapping[expert] = -1
        
        print(f'dropped {num_dropped} experts out of {self.total_experts}')
        print(f'dropped expert norms: {expert_norms[dropped_experts]}')
    
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
                    self.expert_norm[i] = self.expert_norm[i] * (old_num_forwards / self.num_forwards[i]) + expert_out.norm(dim=-1, p=2).mean() * (batch_size / self.num_forwards[i])

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
