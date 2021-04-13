from functools import partial
from typing import Tuple, Callable, List
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from config import ImageSize

Dataset = tf.data.Dataset


def get_datasets(data_dir: Path, img_size: ImageSize, batch_size: int) -> Tuple[Dataset, Dataset, List[str]]:
    dataset_factory = get_base_dataset(data_dir, img_size, batch_size)
    train_dataset = dataset_factory(subset="training")
    validation_dataset = dataset_factory(subset="validation")
    class_names = train_dataset.class_names
    cached_train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    cached_validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return cached_train_dataset, cached_validation_dataset, class_names


def get_base_dataset(data_dir: Path, img_size: ImageSize, batch_size: int) -> Callable[..., tf.data.Dataset]:
    return partial(keras.preprocessing.image_dataset_from_directory,
                   data_dir,
                   validation_split=0.2,
                   seed=123,
                   image_size=img_size,
                   batch_size=batch_size)
