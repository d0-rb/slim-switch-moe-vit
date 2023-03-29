import argparse

from .base import BasePruning


class DropTokens(BasePruning):
    @staticmethod
    def get_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--foo-2", default="Hello")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
