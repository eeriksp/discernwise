from tensorflow.python.keras.callbacks import History

from .conf import TrainingConfig
from .dataset import get_datasets
from .model import get_model


def train(confg: TrainingConfig) -> History:
    train_dataset, validation_dataset, class_names = get_datasets(confg.data_dir, confg.image_size, confg.batch_size)
    model = get_model(confg.image_size, len(class_names))
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=confg.epochs
    )
    model.save(confg.model_path)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(type(acc), type(val_acc), type(loss), type(val_loss))
    return history

# epochs_range = range(EPOCHS)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# model.save(MODEL_PATH)
