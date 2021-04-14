from __future__ import annotations

from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import NamedTuple, List, Tuple
import json


class ImageSize(NamedTuple):
    """
    This tuple format is used by the TensorFlow API.
    This namedtuple is created to make the code more readable.
    """
    height: int
    width: int


@dataclass
class Config:
    """
    The base class for other command-specific configurations
    holding the values required by all the commands.
    """
    img_height: InitVar[int] = 250
    img_width: InitVar[int] = 250
    model_path_str: InitVar[str] = None

    def __post_init__(self, img_height: int, img_width: int, model_path_str: str):
        self.image_size = ImageSize(img_height, img_width)
        self.model_path = Path(model_path_str).resolve()


class ModelConfig:
    """
    Handle persisting and loading DiscernWise specific configuration for each model.
    TensorFlow saves its model as a directory.
    The custom configuration a placed to a file within that directory.
    """
    filename = 'discernwise.json'

    def __init__(self, labels: Tuple[str]):
        self.labels = labels

    def save(self, model_path: Path):
        with open(model_path / self.filename, 'w') as f:
            json.dump({'labels': self.labels}, f)

    @classmethod
    def load(cls, model_path: Path) -> ModelConfig:
        with open(model_path / cls.filename, 'r') as f:
            data = json.load(f)
            return ModelConfig(data['labels'])
