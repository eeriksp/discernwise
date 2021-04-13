from argparse import ArgumentParser

from commands.base import BaseCommand
from presentation.classify import visualize_classification_results
from services.classify import ClassificationConfig, classify


class ClassifyCommand(BaseCommand):
    name = 'classify'
    help = 'classify the given images using the given model'

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('model_path', help='path to the model used for classification')
        parser.add_argument('image_paths', nargs='+', help='paths to the images to be classified')
        parser.add_argument('--gui', action='store_true')

    @staticmethod
    def build_config(args) -> ClassificationConfig:
        return ClassificationConfig(model_path_str=args.model_path, image_str_paths=args.image_paths)

    @classmethod
    def handle(cls, config: ClassificationConfig) -> None:
        results = classify(config)
        visualize_classification_results(results)
