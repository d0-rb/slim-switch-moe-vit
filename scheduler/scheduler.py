import torch as th
from torch.optim.optimizer import Optimizer

from .threshold_scheduler import *


class CurriculumScheduler(object):

    """Enable different curriculum learning curve"""

    def __init__(self, args, writer=None):
        """TODO: to be defined.

        :args: TODO

        """
        self._args = args
        self._writer = writer

        threshold_optimizers = {}
        if args.threshold_scheduler == "cosine":
            threshold_optimizers["dense"] = Optimizer(
                params=[{"params": []}], defaults={"lr": args.starting_threshold_dense}
            )
            threshold_optimizers["moe"] = Optimizer(
                params=[{"params": []}], defaults={"lr": args.starting_threshold_moe}
            )
        elif args.threshold_scheduler == "linear":
            threshold_optimizers["dense"] = Optimizer(
                params=[{"params": []}], defaults={"lr": args.target_threshold_dense}
            )
            threshold_optimizers["moe"] = Optimizer(
                params=[{"params": []}], defaults={"lr": args.target_threshold_moe}
            )

        threshold_schedulers = {}
        if args.threshold_scheduler == "cosine":
            threshold_schedulers["dense"] = CosineAnnealingLRWarmup(
                threshold_optimizers["dense"],
                T_max=args.epochs - 1,
                warmup_steps=args.threshold_warmup_epochs,
                eta_min=args.target_threshold_dense,
                last_epoch=-1,
            )
            threshold_schedulers["moe"] = CosineAnnealingLRWarmup(
                threshold_optimizers["moe"],
                T_max=args.epochs - 1,
                warmup_steps=args.threshold_warmup_epochs,
                eta_min=args.target_threshold_moe,
                last_epoch=-1,
            )
        elif args.threshold_scheduler == "linear":
            threshold_schedulers["dense"] = LinearLRWarmup(
                threshold_optimizers["dense"],
                warmup_steps=args.threshold_warmup_epochs,
                start_factor=args.starting_threshold_dense
                / args.target_threshold_dense,
                end_factor=1.0,
                total_iters=args.epochs,
            )
            threshold_schedulers["moe"] = LinearLRWarmup(
                threshold_optimizers["moe"],
                warmup_steps=args.threshold_warmup_epochs,
                start_factor=args.starting_threshold_moe / args.target_threshold_moe,
                end_factor=1.0,
                total_iters=args.epochs,
            )
        else:
            NotImplementedError

        self.threshold_optimizers = threshold_optimizers
        self.threshold_schedulers = threshold_schedulers

    def _write(self, *args, **kwargs):
        if self._writer is not None:
            self._writer.log_scalar(*args, **kwargs)

    def step(self, epoch, model=None):
        dense = self.threshold_schedulers["dense"].step(epoch)
        self._write("train/thresholds/dense", dense, epoch)
        moe = self.threshold_schedulers["moe"].step(epoch)
        self._write("train/thresholds/gate", moe, epoch)
        if model is not None:
            for name, module in model.named_modules():
                if name.endswith("dense_gate"):
                    module.step(dense[0])
                    self._write(f"threshold/{name}", dense, epoch)
                elif name.endswith("moe_gate"):
                    module.step(moe[0])
                    self._write(f"threshold/{name}", moe, epoch)
        return dense, moe

    def state_dict(self):
        ret = {"threshold_optimizers": {}, "threshold_schedulers": {}}
        for k, v in self.threshold_optimizers.items():
            ret["threshold_optimizers"][k] = v.state_dict()
        for k, v in self.threshold_schedulers.items():
            ret["threshold_schedulers"][k] = v.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        for class_attr in state_dict:
            attr = getattr(self, class_attr)
            for k, v in attr.items():
                v.load_state_dict(state_dict[class_attr][k])
