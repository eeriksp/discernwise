from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import NamedTuple


class ImageSize(NamedTuple):
    height: int
    width: int


@dataclass
class Config:
    batch_size: int = 32
    img_height: InitVar[int] = 250
    img_width: InitVar[int] = 250
    EPOCHS: int = 20
    model_path_str: InitVar[str] = None

    def __post_init__(self, img_height: int, img_width: int, model_path_str: str):
        self.image_size = ImageSize(img_height, img_width)
        self.model_path = Path(model_path_str).resolve()
