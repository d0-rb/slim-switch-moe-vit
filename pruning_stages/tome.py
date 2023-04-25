# pylint: disable=E1101
# mypy: disable-error-code=attr-defined
import argparse
import os
import typing as typ

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from tome.patch.timm import make_tome_class
from tome.patch.timm import ToMeAttention
from tome.patch.timm import ToMeBlock

from .base import BasePruning
from .benchmark import InferenceBenchmarkRunner
from .engine import evaluate
from .engine import train_one_epoch
from .models.vision_transformer import Attention
from .models.vision_transformer import Block


class ToMeDrop(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        # parser.add_argument("--attn-momentum", default=0.75, type=float)
        parser.add_argument("--finetune-epochs", default=10, type=int)
        # parser.add_argument("--top-k", default=2, type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tome_r = np.arange(1, 16, 1)
        # self.tome_r = [3]
        self.init()

    def main(self):
        # evaluate(self.testloader, self.model, self.device)

        acc_b4, acc_af, speed = self.eval(self.tome_r)
        plot(
            acc_b4,
            speed,
            self.tome_r,
            "accuracy",
            "ms/step",
            "acc_b4 vs. speed",
            f"{self.args.output_dir}/acc_b4.pdf",
        )
        plot(
            acc_af,
            speed,
            self.tome_r,
            "accuracy",
            "ms/step",
            "acc_af vs. speed",
            f"{self.args.output_dir}/acc_af.pdf",
        )

    def init(self):
        apply_patch(self.model)
        for name, param in self.model.named_parameters():
            if not "norm" in name:
                if "cls_token" in name:
                    pass
                else:
                    param.requires_grad = False

    def eval(self, tome_r: typ.List[int]):
        acc_b4 = th.zeros(len(tome_r))
        acc_af = th.zeros(len(tome_r))
        speed = th.zeros(len(tome_r))

        org_pth = os.path.join(self.args.output_dir, "orig.pth")
        th.save(self.model.state_dict(), org_pth)

        for i, r in enumerate(tome_r):
            self.model.load_state_dict(th.load(org_pth))
            self.model.r = r
            self.optimizer.__init__(self.model.parameters(), self.args.lr)
            print(f"##################### {r=} ##################")
            throughput = self.benchmark()
            results_b4 = evaluate(self.testloader, self.model, self.device)
            print("fine-tune now")
            # self.finetune()
            print("fine-tunecompleted")
            results_af = evaluate(self.testloader, self.model, self.device)
            acc_b4[i] = results_b4["acc1"]
            # acc_af[i] = results_af["acc1"]
            speed[i] = throughput["step_time"]

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


def plot(x, y, z, xlabel, ylabel, title, filename):
    # Create the line plot
    plt.clf()
    plt.plot(x, y, marker="o", linestyle="-")

    # Annotate the points with the z labels
    for i, label in enumerate(z):
        plt.annotate(
            label, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center"
        )

    # Set labels for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the title of the plot
    plt.title(title)

    # Save the plot to a file
    plt.savefig(filename)

    # Close the plot to release memory
    plt.close()


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.
    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
