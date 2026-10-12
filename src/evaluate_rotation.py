import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "data" / "splits.json"
RESULTS_DIR = PROJECT_DIR / "results"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 32

MODELS = {
    "Baseline CNN": RESULTS_DIR / "baseline_cnn.pth",
    "Augmented CNN": RESULTS_DIR / "augmented_cnn.pth",
}


class WasteDataset(Dataset):
    def __init__(self, records, class_to_idx, rotation=0):
        self.records = records
        self.class_to_idx = class_to_idx
        self.rotation = rotation
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            if self.rotation:
                image = image.rotate(
                    self.rotation, resample=Image.Resampling.BILINEAR
                )

        image = self.transform(image)
        label = self.class_to_idx[record["label"]]
        return image, label


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

    results = {}

    for model_name, model_path in MODELS.items():
        checkpoint = torch.load(
            model_path, map_location=DEVICE, weights_only=True
        )
        classes = checkpoint["classes"]
        class_to_idx = {
            name: i for i, name in enumerate(classes)
        }

        model = WasteCNN(len(classes)).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        results[model_name] = {}

        for angle in [0, 15, 30, 45]:
            dataset = WasteDataset(
                splits["test"], class_to_idx, angle
            )
            loader = DataLoader(
                dataset, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0
            )

            actual, predicted = [], []

            with torch.no_grad():
                for images, labels in loader:
                    outputs = model(images)
                    predicted.extend(outputs.argmax(1).tolist())
                    actual.extend(labels.tolist())

            accuracy = accuracy_score(actual, predicted)
            results[model_name][str(angle)] = accuracy

            print(
                f"{model_name} | Rotation {angle} degrees | "
                f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)"
            )

    output_file = RESULTS_DIR / "rotation_robustness.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
