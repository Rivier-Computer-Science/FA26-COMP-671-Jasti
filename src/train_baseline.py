import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

SPLIT_FILE = Path("data/splits.json")
MODEL_DIR = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class WasteDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as image:
            image = image.convert("RGB")

        if self.transform:
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


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    with SPLIT_FILE.open(encoding="utf-8") as file:
        splits = json.load(file)

    classes = sorted({record["label"] for record in splits["train"]})
    class_to_idx = {name: index for index, name in enumerate(classes)}

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = WasteDataset(
        splits["train"], class_to_idx, transform
    )
    val_dataset = WasteDataset(
        splits["validation"], class_to_idx, transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = WasteCNN(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_accuracy = 0.0
    history = []

    print("Device:", DEVICE)
    print("Classes:", classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "seed": SEED,
            }, MODEL_DIR / "baseline_cnn.pth")

    with (MODEL_DIR / "baseline_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=2)

    print("\nBest validation accuracy:", round(best_val_accuracy, 4))
    print("Saved model: results/baseline_cnn.pth")


if __name__ == "__main__":
    main()