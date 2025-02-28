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
    - images_dir: e.g. /mnt/data/train/images
    - labels_dir: e.g. /mnt/data/train/labels
    - output_split_dir: e.g. /mnt/data/train (where class folders will be created)
    - class_names: list of class names from data.yaml
    """
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Skipping {images_dir} or {labels_dir} because it doesn't exist.")
        return

    # Ensure the output directory exists (this is the parent of images/ and labels/)
    os.makedirs(output_split_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"No label files found in {labels_dir}. Skipping.")
        return
