#!/usr/bin/env python3

import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Environment config
data_dir = os.getenv("DATA_DIR", "/mnt/data")
output_dir = os.getenv("OUTPUT_DIR", "/mnt/model")
epochs = int(os.getenv("EPOCHS", "5"))
batch_size = int(os.getenv("BATCH_SIZE", "32"))
learning_rate = float(os.getenv("LR", "0.001"))

# Transforms: resize to 224x224, apply normalization matching ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Paths for train and val
# Your data-loader job should have created subfolders in /mnt/data/train and /mnt/data/val
# e.g., /mnt/data/train/Aircraft Carrier/, /mnt/data/train/Bulkers/, etc.
data_dir = os.getenv("DATA_DIR", "/mnt/data")

train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                transform=some_transform)
val_ds   = datasets.ImageFolder(os.path.join(data_dir, "valid"),
                                transform=some_transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=..., shuffle=True, ...)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=..., shuffle=False, ...)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

# Optionally create a val dataset/loader if /mnt/data/val exists
if os.path.isdir(val_path):
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)
else:
    val_loader = None

# Confirm how many classes we found
num_classes = len(train_dataset.classes)
print(f"Found {num_classes} classes: {train_dataset.classes}")

# Load a pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Replace the final fully connected layer to match our number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(1, epochs + 1):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    # Average training loss for this epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f}")

    # Validation step (if val_loader exists)
    if val_loader:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = (100.0 * correct / total) if total > 0 else 0.0
        print(f"Validation accuracy: {val_acc:.2f}%")
        model.train()  # back to training mode

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)
model_file = os.path.join(output_dir, "ship_model.pth")

# Save model checkpoint (including class names)
checkpoint = {
    "model_state": model.state_dict(),
    "classes": train_dataset.classes
}
torch.save(checkpoint, model_file)
print(f"Saved trained model to {model_file}")
