

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "microsoft/rad-dino"

print("Loading RAD-DINO model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Model loaded successfully!")

IMAGE_PATH = "sample_xray.png.jpeg"
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

features = outputs.last_hidden_state

print("Image processed successfully!")
print("Feature tensor shape:", tuple(features.shape))
print("Feature extraction completed!")

