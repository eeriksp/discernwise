from typing import Dict

from matplotlib import pyplot as plt
from PIL import Image
from services.classify import ClassificationResult


def visualize_classification_results(results: ClassificationResult) -> None:
    fig = plt.figure(figsize=(10, 10))
    i = 0
    for image_path, labels in results.items():
        sub = fig.add_subplot(1, len(results), i + 1)
        sub.set_title(_compose_labels_str(labels))
        plt.axis('off')
        plt.imshow(Image.open(image_path))
        i += 1
    plt.show()


def _compose_labels_str(labels: Dict[str, float]) -> str:
    lines = [f'{label} {round(probability * 100, 2)}%' for label, probability in labels.items()]
    return '\n'.join(lines)
