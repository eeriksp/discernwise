from pathlib import Path
from typing import Tuple

from tensorflow.python.keras import Model, models

from config import ModelConfig


def save_model(path: Path, model: Model, labels: Tuple[str]) -> None:
    """
    Save the TensorFlow model and custom configuration to the given path.
    """
    model.save(path)
    ModelConfig(labels).save(path)


def load_model(path: Path) -> Tuple[Model, Tuple[str]]:
    """
    Load the TensorFlow model and custom configuration from the given path.
    """
    return models.load_model(path), ModelConfig.load(path).labels
