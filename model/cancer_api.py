from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = torch.device("cpu")
app = FastAPI(title="Cancer Detection API")

# Load checkpoint
checkpoint = torch.load("best_cv_model.pkl", map_location=DEVICE)

class_names = checkpoint["class_names"]
IMG_SIZE = checkpoint["img_size"]

# Rebuild model
model = models.densenet169(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model=model.eval()

# SAME transform as validation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = class_names[predicted.item()]

    return {
        "prediction": prediction,
        "confidence": float(confidence.item())
    }