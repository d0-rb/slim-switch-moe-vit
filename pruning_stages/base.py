import argparse
import typing as typ

import torch as th


class BasePruning:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        pass

    def main(self):
        self.prune()
        self.finetune()

    def finetune(self, *args, **kwargs):
        pass

    def prune(self, *args, **kwargs):
        pass
