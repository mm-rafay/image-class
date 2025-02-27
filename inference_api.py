import os, json
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms, models

app = FastAPI()

# Load model checkpoint and classes at startup
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/model/ship_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transforms as used during training for input preprocessing
input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the saved model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["classes"]
num_classes = len(class_names)

# Recreate the model architecture and load state dict
model = models.resnet50(weights=None, num_classes=num_classes)  # no pretraining weights needed for structure
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file and prepare it
    image = Image.open(file.file).convert("RGB")
    tensor = input_transform(image).unsqueeze(0).to(device)  # transform and create batch of size 1
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # get probabilities for the single batch item
    # Get top prediction
    top_idx = torch.argmax(probs, dim=0).item()
    pred_class = class_names[top_idx]
    confidence = probs[top_idx].item()
    return {"predicted_class": pred_class, "confidence": round(confidence, 4)}
