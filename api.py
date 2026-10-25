

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
import torch
import io
import time
app = FastAPI(title="MedLens AI API")

# Load the AI model once when the API starts
MODEL_NAME = "microsoft/rad-dino"

print("Loading RAD-DINO model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("RAD-DINO model loaded successfully!")
@app.get("/")
def home():
    return {
        "message": "Welcome to MedLens AI API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "project": "MedLens AI",
        "api_status": "Online",
        "model": "Microsoft RAD-DINO",
        "model_status": "Loaded",
        "purpose": "Medical image feature extraction",
        "diagnosis_enabled": False
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image."
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    try:
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        features = outputs.last_hidden_state

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="AI image processing failed."
        ) from error

    processing_time = round(time.perf_counter() - start_time, 2)

    return {
        "filename": file.filename,
        "message": "Image processed successfully.",
        "analysis_status": "RAD-DINO feature extraction completed",
        "processing_time_seconds": processing_time,
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "feature_tensor_shape": list(features.shape),
        "feature_count": int(features.shape[1]),
        "feature_dimension": int(features.shape[2]),
        "medical_diagnosis": "Not provided"
    }



