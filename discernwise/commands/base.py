from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import final

from config import Config

root_parser = ArgumentParser()
subparsers = root_parser.add_subparsers(help='top-level commands', dest='command')


class BaseCommand(ABC):
    name: str = None  # must be overridden in subclass
    help: str = None  # must be overridden in subclass

    def __init_subclass__(cls):
        command_parser = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_arguments(command_parser)

    @staticmethod
    @abstractmethod
    def add_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_config(args) -> Config:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def handle(cls, config: Config) -> None:
        raise NotImplementedError()

    @final
    @classmethod
    def execute(cls) -> None:
        args = root_parser.parse_args()
        command_handler = [h for h in cls.__subclasses__() if h.name == args.command][0]
        command_handler.handle(command_handler.build_config(args))
