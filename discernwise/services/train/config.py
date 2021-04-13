from dataclasses import dataclass, InitVar
from pathlib import Path

from config import Config


@dataclass
class TrainingConfig(Config):
    epochs: int = 2
    batch_size: int = 32
    data_dir_str: InitVar[str] = None

    def __post_init__(self, img_height: int, img_width: int, model_path_str: str, data_dir_str: str):
        super().__post_init__(img_height, img_width, model_path_str)
        self.data_dir = Path(data_dir_str).resolve()
