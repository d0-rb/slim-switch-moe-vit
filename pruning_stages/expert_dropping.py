import argparse

from .base import BasePruning


class ExpertDropping(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--foo-1", default="Hello")
        # add your specific argument here #

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
