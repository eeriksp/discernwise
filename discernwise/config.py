from dataclasses import dataclass, InitVar
from pathlib import Path


@dataclass
class Config:
    batch_size: int = 32
    img_height: int = 250
    img_width: int = 250
    EPOCHS: int = 20
    model_path_str: InitVar[str] = None

    def __post_init__(self, model_path_str: str):
        self.model_path = Path(model_path_str).resolve()
