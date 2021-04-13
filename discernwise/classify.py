import numpy as np

import tensorflow as tf
from tensorflow import keras

evaluated_image_path = 'my-cup.jpg'

MODEL_PATH = 'model'
IMG_HEIGHT = 250
IMG_WIDTH = 250
CLASS_NAMES = ['cardboard', 'coffee cup', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = tf.keras.models.load_model(MODEL_PATH)

img = keras.preprocessing.image.load_img(evaluated_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

prediction = model.predict(img_array)[0]

score = tf.nn.softmax(prediction)
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
)
