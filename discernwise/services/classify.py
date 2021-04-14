from collections import OrderedDict
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import List, Dict

import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import Config, ModelConfig
from utils.collections import sort_by_values


@dataclass
class ClassificationConfig(Config):
    image_str_paths: InitVar[List[str]] = None

    def __post_init__(self, img_height: int, img_width: int, model_path_str: str, image_str_paths: [List[str]]):
        super().__post_init__(img_height, img_width, model_path_str)
        self.image_paths = [Path(img).resolve() for img in image_str_paths]


"""
A dictionary mapping the path of the image to the classification outcome.
The classification outcome is in turn a dictionary mapping the labels to the probabilities.
"""
ClassificationResult = Dict[Path, Dict[str, float]]


def classify(config: ClassificationConfig) -> ClassificationResult:
    """
    Classify the given images using the given model and
    return the probabilities that any given image matches any given label.
    """
    model = tf.keras.models.load_model(config.model_path)
    result = OrderedDict()
    class_names = ModelConfig.load(config.model_path).class_names
    for img_path in config.image_paths:
        img = tf.expand_dims(image.img_to_array(image.load_img(img_path, target_size=config.image_size)), 0)
        prediction = model.predict(img)[0]
        np_scores = tf.nn.softmax(prediction)
        scores = [float(i) for i in np_scores]
        result[img_path] = OrderedDict()
        for i in range(len(class_names)):
            result[img_path][class_names[i]] = scores[i]
            result[img_path] = sort_by_values(result[img_path], reverse=True)
    return result
