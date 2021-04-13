from commands import *

BaseCommand.execute()

# import argparse
#
# p = argparse.ArgumentParser()
# subparsers = p.add_subparsers(help='top-level commands', dest='command')
#
# train_parser = subparsers.add_parser('train', help='train a new model with the given dataset')
# train_parser.add_argument('dataset_path',
#                           help='path to the dataset directory containing a subdirectory for each category')
# train_parser.add_argument('model_path', help='path where to save the new trained model')
#
# classify_parser = subparsers.add_parser('classify', help='classify the given image using the given model')
# classify_parser.add_argument('image_path', help='path to the image we want to classify')
# classify_parser.add_argument('model_path', help='path to the model used for classification')
#
# print(p.parse_args())
#
# class A:
#
#     def __init_subclass__(cls, **kwargs):
#         print('Hello!')
#         cls().foo()
#
# class B(A):
#     def foo(self):
#         print('Ohhoo')
