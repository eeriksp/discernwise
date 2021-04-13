from argparse import ArgumentParser

from commands.base import BaseCommand
from presentation.train import visualize_training_results
from services.train import TrainingConfig, train


class TrainCommand(BaseCommand):
    name = 'train'
    help = 'train a new model with the given dataset'

    @staticmethod
    def add_arguments(p: ArgumentParser) -> None:
        p.add_argument('model_path', help='path where to save the new trained model')
        p.add_argument('dataset_path', help='path to the dataset directory containing a subdirectory for each category')
        p.add_argument('--epochs', type=int, default=2, dest="epochs", help='The number of epochs used for training')

    @staticmethod
    def build_config(args) -> TrainingConfig:
        return TrainingConfig(model_path_str=args.model_path, data_dir_str=args.dataset_path, epochs=args.epochs)

    @classmethod
    def handle(cls, config: TrainingConfig) -> None:
        results = train(config)
        visualize_training_results(results)
