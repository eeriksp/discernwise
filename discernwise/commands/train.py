from argparse import ArgumentParser
from dataclasses import dataclass, InitVar
from pathlib import Path

from commands.base import BaseCommand
from config import Config


@dataclass
class TrainingConfig(Config):
    epochs: int = 2
    data_dir_str: InitVar[str] = None

    def __post_init__(self, model_path_str: str, data_dir_str: str):
        super().__post_init__(model_path_str)
        self.data_dir = Path(data_dir_str).resolve()


class TrainCommand(BaseCommand):
    name = 'train'
    help = 'train a new model with the given dataset'

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('dataset_path',
                            help='path to the dataset directory containing a subdirectory for each category')
        parser.add_argument('model_path', help='path where to save the new trained model')

    @staticmethod
    def build_config(args) -> TrainingConfig:
        return TrainingConfig(model_path_str=args.model_path, data_dir_str=args.dataset_path)

    @classmethod
    def handle(cls, config: TrainingConfig) -> None:
        print(f'Here is our config: {config}')
        print(config.model_path)
        print(config.data_dir)
