from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
import time
import torch
from transformers import AutoImageProcessor, AutoModel
app = FastAPI(title="MedLens AI API")
MODEL_NAME = "microsoft/rad-dino"

processor = None
model = None

def load_model():
    global processor, model

    if model is None:
        print("Loading RAD-DINO model...")

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()

        print("RAD-DINO model loaded successfully!")

@app.get("/")
def home():
    return {"message": "MedLens AI API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image."
        )

    start = time.time()
    content = await file.read()

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    try:
        load_model()

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        features = outputs.last_hidden_state
        feature_vector = features.mean(dim=1).squeeze().tolist()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAD-DINO processing failed: {str(e)}"
        )

    return {
        "filename": file.filename,
        "image_size": list(image.size),
        "analysis_status": "RAD-DINO feature extraction successful",
        "model": MODEL_NAME,
        "feature_shape": list(features.shape),
        "feature_vector_length": len(feature_vector),
        "feature_preview": feature_vector[:10],
        "processing_time_seconds": round(time.time() - start, 3),
        "medical_diagnosis": "Not provided"
    }

