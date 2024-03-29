from argparse import ArgumentParser

from commands.base import BaseCommand
from presentation.classify import display_classification_results
from services.classify import ClassificationConfig, classify


class ClassifyCommand(BaseCommand):
    """
    Use the given model to classify the given images (one or more).
    The results will be displayed as a GUI window.
    """
    name = 'classify'
    help = 'classify the given images using the given model'

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('model_path', help='path to the model used for classification')
        parser.add_argument('image_paths', nargs='+', help='paths to the images to be classified')

    @staticmethod
    def build_config(args) -> ClassificationConfig:
        return ClassificationConfig(model_path_str=args.model_path, image_str_paths=args.image_paths)

    @staticmethod
    def handle(config: ClassificationConfig) -> None:
        results = classify(config)
        display_classification_results(results)
