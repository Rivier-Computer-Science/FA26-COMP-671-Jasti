from pathlib import Path
from PIL import Image

DATASET_PATH = Path(
    r"S:\AI Document\deep learning\project\realwaste\realwaste-main\RealWaste"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0

for category in sorted(DATASET_PATH.iterdir()):
    if not category.is_dir():
        continue

    image_files = [
        file for file in category.rglob("*")
        if file.suffix.lower() in VALID_EXTENSIONS
    ]

    valid_images = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images += 1
        except Exception:
            print(f"Unreadable image: {image_path}")

    total_images += valid_images
    print(f"{category.name}: {valid_images} images")

print(f"\nTotal valid images: {total_images}")