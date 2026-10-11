import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Configuration
DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 32

SPLIT_FILE = Path("data/splits.json")
MODEL_FILE = Path("results/baseline_cnn.pth")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# Dataset loader
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


# Same CNN architecture used during training
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
    # Load saved test split
    with SPLIT_FILE.open(encoding="utf-8") as file:
        splits = json.load(file)

    # Load the trained model
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
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

    # Predict without updating model weights
    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    # Calculate evaluation metrics
    accuracy = accuracy_score(actual, predicted)

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    print("\nBASELINE CNN — TEST RESULTS")
    print("=" * 40)
    print(f"Test images: {len(actual)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))

    # Save numerical results
    results = {
        "model": "Baseline CNN",
        "test_images": len(actual),
        "accuracy": float(accuracy),
        "classification_report": report,
        "classes": classes,
    }

    with (RESULTS_DIR / "baseline_test_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    # Create confusion matrix figure
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(classes))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Baseline CNN — Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "baseline_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved results:")
    print("results/baseline_test_results.json")
    print("figures/baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()