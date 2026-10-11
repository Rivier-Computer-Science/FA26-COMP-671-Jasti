
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "data" / "splits.json"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# --------------------------------------------------
# 3. Dataset loader
# --------------------------------------------------

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

        label = self.class_to_idx[record["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# 4. CNN model
# --------------------------------------------------

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
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


# --------------------------------------------------
# 5. Load the existing data split
# --------------------------------------------------

with open(SPLIT_FILE, "r", encoding="utf-8") as file:
    split_data = json.load(file)

train_records = split_data["train"]
val_records = split_data["validation"]

classes = sorted({
    record["label"] for record in train_records
})
class_to_idx = {
    name: index for index, name in enumerate(classes)
}
num_classes = len(classes)

print(f"Training images: {len(train_records)}")
print(f"Validation images: {len(val_records)}")
print(f"Number of classes: {num_classes}")

train_dataset = WasteDataset(
    train_records,
    class_to_idx,
    transform=train_transform
)

val_dataset = WasteDataset(
    val_records,
    class_to_idx,
    transform=eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 6. Initialize model and optimizer
# --------------------------------------------------

model = WasteCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_accuracy = 0.0
history = []


# --------------------------------------------------
# 7. Train and validate
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    average_train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    average_val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": average_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": average_val_loss,
        "val_accuracy": val_accuracy
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {average_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "seed": SEED
            },
            RESULTS_DIR / "augmented_cnn.pth"
        )


# --------------------------------------------------
# 8. Save training history
# --------------------------------------------------

with open(
    RESULTS_DIR / "augmented_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(history, file, indent=4)

print("\nAugmented CNN training complete.")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Model saved to: {RESULTS_DIR / 'augmented_cnn.pth'}")
print(f"History saved to: {RESULTS_DIR / 'augmented_history.json'}")


