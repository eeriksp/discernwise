from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import final

from config import Config

root_parser = ArgumentParser()
subparsers = root_parser.add_subparsers(help='top-level commands', dest='command')


class BaseCommand(ABC):
    """
    The base of all commands.
    A command is a standalone unit of work that the program will do and will exit upon outputing the results.
    Commands are called as the first argument for the program
    (e.g. the `train` command is called as `discernwise train [options]`).

    The define a new command
      1. create a new file in the `commands` directory
      2. create a subclass of this class in the newly created file
      3. import it to `commands/__init__.py`
    """
    name: str = None  # the name of the command used to invoke it from the CLI; must be overridden in subclasses
    help: str = None  # the help message displayed to the user in the CLI; must be overridden in subclass

    def __init_subclass__(cls):
        command_parser = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_arguments(command_parser)

    @staticmethod
    @abstractmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """
        Use `parser.add_argument()` to specify the arguments for the command.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_config(args) -> Config:
        """
        :param args: The arguments specified in `self.add_arguments()` as returned by `parser.parse_args()`.
        :return: The configuration which is then passed to `self.handle()`.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def handle(config: Config) -> None:
        """
        :param config: The config obtained from `self.build_config()`.
        This method should take care of performing the intended action of the command
        and outputting the results (as a GUI window, JSON to standard out or in any other way).
        """
        raise NotImplementedError()

    @final
    @classmethod
    def execute(cls) -> None:
        args = root_parser.parse_args()
        command_handler = [h for h in cls.__subclasses__() if h.name == args.command][0]
        command_handler.handle(command_handler.build_config(args))
