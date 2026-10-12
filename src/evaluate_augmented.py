import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "data" / "splits.json"
MODEL_FILE = PROJECT_DIR / "results" / "augmented_cnn.pth"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 32


class WasteDataset(Dataset):
    def __init__(self, records, class_to_idx, transform):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = self.class_to_idx[record["label"]]

        return image, label


class WasteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with SPLIT_FILE.open(encoding="utf-8") as file:
        splits = json.load(file)

    checkpoint = torch.load(
        MODEL_FILE, map_location=DEVICE, weights_only=True
    )

    classes = checkpoint["classes"]
    class_to_idx = {
        name: index for index, name in enumerate(classes)
    }

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    test_dataset = WasteDataset(
        splits["test"], class_to_idx, transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = WasteCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())

    accuracy = accuracy_score(actual, predicted)

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(classes)))
    )

    output = {
        "model": "Augmented CNN",
        "test_images": len(test_dataset),
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    results_file = RESULTS_DIR / "augmented_test_results.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Augmented CNN - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "augmented_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nAugmented CNN Test Results")
    print("--------------------------")
    print("Test images:", len(test_dataset))
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))
    print("Results saved to:", results_file)
    print("Confusion matrix saved to:",
          FIGURES_DIR / "augmented_confusion_matrix.png")


if __name__ == "__main__":
    main()
