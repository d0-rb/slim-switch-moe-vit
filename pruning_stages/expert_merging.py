import argparse

from .base import BasePruning


class ExpertMerging(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--foo-3", default="Hello")
        # add your specific argument here #

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
