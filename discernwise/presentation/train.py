from matplotlib import pyplot as plt

from services.train.train import TrainingResults


def visualize_training_results(results: TrainingResults) -> None:
    epochs_range = range(results.epochs_count)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, results.accuracy, label='Training Accuracy')
    plt.plot(epochs_range, results.validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, results.loss, label='Training Loss')
    plt.plot(epochs_range, results.validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
