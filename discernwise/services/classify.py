from collections import OrderedDict
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import Config, ModelConfig


@dataclass
class ClassificationConfig(Config):
    image_str_paths: InitVar[List[str]] = None

    def __post_init__(self, img_height: int, img_width: int, model_path_str: str, image_str_paths: [List[str]]):
        super().__post_init__(img_height, img_width, model_path_str)
        self.image_paths = [Path(img).resolve() for img in image_str_paths]


def classify(config: ClassificationConfig) -> Dict[Path, Dict[str, float]]:
    model = tf.keras.models.load_model(config.model_path)
    result = OrderedDict()
    class_names = ModelConfig.load(config.model_path).class_names
    for img_path in config.image_paths:
        img = tf.expand_dims(image.img_to_array(image.load_img(img_path, target_size=config.image_size)), 0)
        prediction = model.predict(img)[0]
        score = tf.nn.softmax(prediction)
        result[img_path] = OrderedDict()
        print(score)
        result[img_path][class_names[np.argmax(score)]] = 100 * np.max(score)
    return result
#
# import numpy as np
#
# import tensorflow as tf
# from tensorflow import keras
#
# evaluated_image_path = 'my-cup.jpg'
#
# MODEL_PATH = 'model'
# IMG_HEIGHT = 250
# IMG_WIDTH = 250
# CLASS_NAMES = ['cardboard', 'coffee cup', 'glass', 'metal', 'paper', 'plastic', 'trash']
#
# model = tf.keras.models.load_model(MODEL_PATH)
#
# img = keras.preprocessing.image.load_img(evaluated_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
# prediction = model.predict(img_array)[0]
#
# score = tf.nn.softmax(prediction)
# print(score)
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
# )
