from pathlib import Path
import random
import shutil

# =============================
# Configuration
# =============================
CONFIG = {
    # Original folder exported from Label Studio in YOLO format
    "source_dir": "in_out_data/labelstudio_yolo/TRG_containers",
    
    # Target folder compatible with Ultralytics
    "output_dir": "in_out_data/datasets/TRG_containers",
    
    # Recommended split for ~120 images
    "split_ratios": {
        "train": 0.75,
        "valid": 0.17,
        "test": 0.08
    }
}

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def main():
    source_dir = Path(CONFIG["source_dir"])
    out_dir = Path(CONFIG["output_dir"])

    source_images = source_dir / "images"
    source_labels = source_dir / "labels"
    classes_file = source_dir / "classes.txt"

    assert source_images.exists(), f"Directory does not exist: {source_images}"
    assert source_labels.exists(), f"Directory does not exist: {source_labels}"
    assert classes_file.exists(), f"File does not exist: {classes_file}"

    # Read classes
    classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Find images
    images = [p for p in source_images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    ratios = CONFIG["split_ratios"]
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["valid"])

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    # Create folders
    for split in splits:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy images and labels
    for split, split_images in splits.items():
        for img_path in split_images:
            label_path = source_labels / f"{img_path.stem}.txt"

            shutil.copy2(img_path, out_dir / split / "images" / img_path.name)

            if label_path.exists():
                shutil.copy2(label_path, out_dir / split / "labels" / label_path.name)
            else:
                # Image without objects: empty label
                (out_dir / split / "labels" / f"{img_path.stem}.txt").write_text("")

    # Create data.yaml
    names_yaml = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])

    data_yaml = f"""path: {out_dir.resolve()}
train: train/images
val: valid/images
test: test/images

names:
{names_yaml}
"""

    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Dataset created at: {out_dir.resolve()}")
    print(f"Train: {len(splits['train'])}")
    print(f"Valid: {len(splits['valid'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()