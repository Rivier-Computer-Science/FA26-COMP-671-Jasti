
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "data" / "splits.json"
MODEL_FILE = ROOT / "results" / "baseline_cnn.pth"
FIGURES_DIR = ROOT / "figures"

IMAGE_SIZE = 128
DEVICE = torch.device("cpu")
MAX_IMAGES = 12


class WasteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    with SPLIT_FILE.open(encoding="utf-8") as file:
        splits = json.load(file)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    model = WasteCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mistakes = []

    for record in splits["test"]:
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            predicted_idx = int(probabilities.argmax().item())

        true_label = record["label"]
        predicted_label = classes[predicted_idx]

        if predicted_label != true_label:
            mistakes.append({
                "path": record["path"],
                "true": true_label,
                "predicted": predicted_label,
                "confidence": float(probabilities[predicted_idx]),
            })

    print(f"Test images: {len(splits['test'])}")
    print(f"Misclassified images: {len(mistakes)}")
    print(f"Test accuracy: {(1 - len(mistakes) / len(splits['test'])) * 100:.2f}%")

    if not mistakes:
        print("No misclassified images found.")
        return

    # Save all errors for later analysis.
    results_file = ROOT / "results" / "baseline_misclassified.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(mistakes, file, indent=2)

    # Display up to 12 examples, starting with the highest confidence
    # incorrect predictions.
    examples = sorted(
        mistakes, key=lambda item: item["confidence"], reverse=True
    )[:MAX_IMAGES]

    columns = 3
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, item in zip(axes, examples):
        with Image.open(item["path"]) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            f"True: {item['true']}\n"
            f"Predicted: {item['predicted']}\n"
            f"Confidence: {item['confidence']:.1%}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(examples):]:
        ax.axis("off")

    fig.suptitle("Baseline CNN: Misclassified Test Images", fontsize=15)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    output = FIGURES_DIR / "baseline_misclassified_examples.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output.relative_to(ROOT)}")
    print(f"Saved: {results_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
