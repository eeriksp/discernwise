from functools import partial
from typing import Tuple, Callable, List
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from config import ImageSize

Dataset = tf.data.Dataset


def get_datasets(data_dir: Path, img_size: ImageSize, batch_size: int) -> Tuple[Dataset, Dataset, Tuple[str]]:
    """
    :return: A tuple containing
      1. the training dataset,
      2. the validation dataset
      3. the labels in alphabetical order
    """
    dataset_factory = _get_dataset_factory(data_dir, img_size, batch_size)
    train_dataset = dataset_factory(subset="training")
    validation_dataset = dataset_factory(subset="validation")
    labels = tuple(train_dataset.class_names)
    cached_train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    cached_validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return cached_train_dataset, cached_validation_dataset, labels


def _get_dataset_factory(data_dir: Path, img_size: ImageSize, batch_size: int) -> Callable[..., tf.data.Dataset]:
    """
    Return a `partial` object, which can be called to obtain the datasets:
        train_dataset = dataset_factory(subset="training")
        validation_dataset = dataset_factory(subset="validation")
    where `dataset_factory` is the return value of this function.
    :param data_dir: path of the directory with the training data
    """
    return partial(keras.preprocessing.image_dataset_from_directory,
                   data_dir,
                   validation_split=0.2,
                   seed=123,
                   image_size=img_size,
                   batch_size=batch_size)
