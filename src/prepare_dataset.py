from pathlib import Path
from sklearn.model_selection import train_test_split
import json

DATASET_PATH = Path(
    r"S:\AI Document\deep learning\project\realwaste\realwaste-main\RealWaste"
)

OUTPUT_PATH = Path("data/splits.json")
SEED = 42

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = []
labels = []

for category in sorted(DATASET_PATH.iterdir()):
    if not category.is_dir():
        continue

    for image_path in sorted(category.rglob("*")):
        if image_path.suffix.lower() in valid_extensions:
            images.append(str(image_path.resolve()))
            labels.append(category.name)

# 70% training, 30% temporary
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

# Split temporary data equally into validation and test
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels,
)

splits = {
    "train": list(zip(train_images, train_labels)),
    "validation": list(zip(val_images, val_labels)),
    "test": list(zip(test_images, test_labels)),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        {
            name: [
                {"path": path, "label": label}
                for path, label in entries
            ]
            for name, entries in splits.items()
        },
        file,
        indent=2,
    )

for name, entries in splits.items():
    print(f"{name}: {len(entries)} images")

print(f"\nSaved split manifest to: {OUTPUT_PATH}")