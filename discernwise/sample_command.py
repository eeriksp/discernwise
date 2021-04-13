from argparse import ArgumentParser

from commands.base import BaseCommand


class SampleCommand(BaseCommand):
    name = 'samplecommand'
    help = 'sample command'

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument('model_path', help='path where to save the new trained model')


class OtherCommand(BaseCommand):
    name = 'othercommand'
    help = 'other command'

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument('anotherarg', help='another argument')
