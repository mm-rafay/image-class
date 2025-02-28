#!/usr/bin/env python3

import os
import shutil
import yaml

# Path to your data root
DATA_ROOT = "/mnt/data"

# Where your YOLO-style train images and labels live:
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABELS_DIR = os.path.join(DATA_ROOT, "train", "labels")

# Where your YOLO-style valid images and labels live:
VALID_IMAGES_DIR = os.path.join(DATA_ROOT, "valid", "images")
VALID_LABELS_DIR = os.path.join(DATA_ROOT, "valid", "labels")

# Path to data.yaml (assumed to be in /mnt/data)
DATA_YAML_PATH = os.path.join(DATA_ROOT, "data.yaml")


def reorganize_split(images_dir, labels_dir, output_split_dir, class_names):
    """
    Convert YOLO-labeled images into ImageFolder format.
      images_dir: e.g. /mnt/data/train/images
      labels_dir: e.g. /mnt/data/train/labels
      output_split_dir: e.g. /mnt/data/train (where class folders will be created)
      class_names: list of class names from data.yaml
    """
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Skipping {images_dir} or {labels_dir} because it doesn't exist.")
        return

    # Ensure the output directory exists
    os.makedirs(output_split_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"No label files found in {labels_dir}. Skipping.")
        return

    # Include common image extensions (both lowercase and uppercase)
    candidate_exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            continue  # Skip label files with no bounding boxes

        # YOLO format: "class_id cx cy w h"
        first_line_parts = lines[0].split()
        try:
            class_id = int(first_line_parts[0])
        except ValueError:
            print(f"Skipping {label_file}. First token is not an integer.")
            continue

        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class{class_id}"

        # Get the base name of the label file (without .txt extension)
        base_name = os.path.splitext(label_file)[0]
        image_path = None
        for ext in candidate_exts:
            test_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break

        if not image_path:
            print(f"WARNING: No matching image found for {label_file} in {images_dir}")
            continue

        # Create subfolder for the class in the output directory
        class_dir = os.path.join(output_split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Copy the image to the class subfolder
        dest_image_path = os.path.join(class_dir, os.path.basename(image_path))
        print(f"Copying {image_path} -> {dest_image_path}")
        shutil.copy2(image_path, dest_image_path)


def cleanup_raw_data(split):
    """
    Remove the original 'images' and 'labels' directories for a given split (e.g., "train" or "valid").
    """
    for sub in ["images", "labels"]:
        path_to_remove = os.path.join(DATA_ROOT, split, sub)
        if os.path.exists(path_to_remove):
            shutil.rmtree(path_to_remove)
            print(f"Removed {path_to_remove}")


def main():
    if not os.path.exists(DATA_YAML_PATH):
        print(f"ERROR: {DATA_YAML_PATH} does not exist. Please correct the path.")
        return

    with open(DATA_YAML_PATH, "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get("names", [])
    print("Loaded class names from data.yaml:", class_names)

    print("\nReorganizing TRAIN set...\n")
    reorganize_split(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, os.path.join(DATA_ROOT, "train"), class_names)

    print("\nReorganizing VALID set...\n")
    reorganize_split(VALID_IMAGES_DIR, VALID_LABELS_DIR, os.path.join(DATA_ROOT, "valid"), class_names)

    # Cleanup the raw directories so only class folders remain
    cleanup_raw_data("train")
    cleanup_raw_data("valid")

    print("\nDone. /mnt/data/train and /mnt/data/valid now contain only class subdirectories.")
    print("You can point ImageFolder to /mnt/data/train or /mnt/data/valid.")


if __name__ == "__main__":
    main()
