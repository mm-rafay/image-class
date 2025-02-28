#!/usr/bin/env python3

import os
import shutil
import yaml

# Path to your data root
DATA_ROOT = "/mnt/data"

# Where your YOLO-style train images and labels live:
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABELS_DIR = os.path.join(DATA_ROOT, "train", "labels")

# Where your YOLO-style val images and labels live:
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "valid", "images")
VAL_LABELS_DIR = os.path.join(DATA_ROOT, "valid", "labels")

# We'll assume data.yaml is in /mnt/data or /mnt/data/train
# Adjust if it's in a different place
DATA_YAML_PATH = os.path.join(DATA_ROOT, "train", "data.yaml")


def reorganize_split(images_dir, labels_dir, output_split_dir, class_names):
    """
    Convert YOLO-labeled images into ImageFolder format.
      images_dir: e.g. /mnt/data/train/images
      labels_dir: e.g. /mnt/data/train/labels
      output_split_dir: e.g. /mnt/data/train
      class_names: list of class names from data.yaml
    """
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Skipping {images_dir} or {labels_dir} because it doesn't exist.")
        return

    # Make sure the output dir exists
    os.makedirs(output_split_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"No label files found in {labels_dir}. Skipping.")
        return

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            # No bounding boxes => skip
            continue

        # Example line: "7 0.5 0.5 1 1"
        first_line_parts = lines[0].split()
        class_id_str = first_line_parts[0]
        try:
            class_id = int(class_id_str)
        except ValueError:
            print(f"Skipping {label_file}. First token '{class_id_str}' isn't an integer.")
            continue

        # Map class ID -> class name from data.yaml, or a fallback
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class{class_id}"

        # Figure out the matching image filename
        # e.g. label_file = "example_123.txt" => image "example_123.jpg"
        base_name = os.path.splitext(label_file)[0]
        # Common image extensions
        candidate_exts = [".jpg", ".jpeg", ".png"]
        image_path = None
        for ext in candidate_exts:
            try_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(try_path):
                image_path = try_path
                break

        if not image_path:
            print(f"WARNING: No matching image found for {label_file}")
            continue

        # Create subfolder for this class
        class_dir = os.path.join(output_split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Copy the image there
        dest_image_path = os.path.join(class_dir, os.path.basename(image_path))
        print(f"Copying {image_path} -> {dest_image_path}")
        shutil.copy2(image_path, dest_image_path)


def main():
    # 1) Load class names from data.yaml
    if not os.path.exists(DATA_YAML_PATH):
        print(f"ERROR: {DATA_YAML_PATH} does not exist. Please correct the path.")
        return

    with open(DATA_YAML_PATH, "r") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", [])
    print("Loaded class names from data.yaml:\n", class_names)

    # 2) Reorganize train
    print("\nReorganizing TRAIN set...\n")
    reorganize_split(
        images_dir=TRAIN_IMAGES_DIR,
        labels_dir=TRAIN_LABELS_DIR,
        output_split_dir=os.path.join(DATA_ROOT, "train"),  # e.g. /mnt/data/train
        class_names=class_names
    )

    # 3) Reorganize val
    print("\nReorganizing VAL set...\n")
    reorganize_split(
        images_dir=VAL_IMAGES_DIR,
        labels_dir=VAL_LABELS_DIR,
        output_split_dir=os.path.join(DATA_ROOT, "valid"),  # e.g. /mnt/data/valid
        class_names=class_names
    )

    print("\nDone. Now your /mnt/data/train and /mnt/data/valid should have subfolders named by class.\n")
    print("You can point ImageFolder to /mnt/data/train or /mnt/data/valid.")


if __name__ == "__main__":
    main()
