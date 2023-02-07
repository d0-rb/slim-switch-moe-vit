import typing as typ

import torch as th


class BasePruning:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        args,
        writer,
        **kwargs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.args = args
        self.writer = writer

    def main(self):
        self.prune()
        self.finetune()

    def finetune(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, *args, **kwargs):
        raise NotImplementedError
