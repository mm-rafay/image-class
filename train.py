import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Config from environment
data_dir = os.getenv("DATA_DIR", "/mnt/data")
output_dir = os.getenv("OUTPUT_DIR", "/mnt/model")
epochs = int(os.getenv("EPOCHS", "5"))
batch_size = int(os.getenv("BATCH_SIZE", "32"))
learning_rate = float(os.getenv("LR", "0.001"))

# Prepare data transformations (resize images and normalize as per ImageNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # resize all images to 224x224
    transforms.RandomHorizontalFlip(),        # simple augmentation
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Dataset and DataLoader
train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "val")       # assume a 'val' subdirectory for validation data
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
if os.path.isdir(val_path):
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
else:
    val_loader = None

num_classes = len(train_dataset.classes)
print(f"Found {num_classes} classes: {train_dataset.classes}")

# Load a pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # pretrained on ImageNet
# Replace the final fully connected layer to match our classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optionally, freeze earlier layers for transfer learning (here we fine-tune entire model)
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(1, epochs+1):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Training loss: {epoch_loss:.4f}")
    # If validation set is available, evaluate performance
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
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"Validation accuracy: {val_acc:.2f}%")
        model.train()  # back to training mode

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)
model_file = os.path.join(output_dir, "ship_model.pth")

# Save model checkpoint (state dict and class names)
checkpoint = {
    "model_state": model.state_dict(),
    "classes": train_dataset.classes
}
torch.save(checkpoint, model_file)
print(f"Saved trained model to {model_file}")
