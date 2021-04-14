from typing import NamedTuple, List
import json

from config import ModelConfig
from model import save_model
from .config import TrainingConfig
from .dataset import get_datasets
from .model import get_model


class TrainingResults(NamedTuple):
    accuracy: List[float]
    validation_accuracy: List[float]
    loss: List[float]
    validation_loss: List[float]

    @property
    def epochs_count(self) -> int:
        """
        Returns the number or epochs.
        All of the four lists should have the same length,
        so `accuracy` is just a random choice.
        """
        return len(self.accuracy)


def train(confg: TrainingConfig) -> TrainingResults:
    """
    Train a new model based on the `config`, save it to the disk and return the training statistics.
    """
    train_dataset, validation_dataset, labels = get_datasets(confg.data_dir, confg.image_size, confg.batch_size)
    model = get_model(confg.image_size, len(labels))
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=confg.epochs
    ).history
    save_model(confg.model_path, model, labels)
    return TrainingResults(history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
