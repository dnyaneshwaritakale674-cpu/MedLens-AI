
# 🩺 MedLens AI

### AI-Powered Medical Image Feature Extraction

MedLens AI is a student project that uses a pretrained RAD-DINO model to extract visual features from uploaded X-ray images.

## 🚀 Features

- Upload PNG and JPEG images
- Process images using a pretrained RAD-DINO model
- Extract image features using a FastAPI backend
- Display feature information in a Streamlit dashboard
- Show image dimensions and processing time
- Download analysis results as a PDF report

## 🛠️ Technologies Used

- Python
- Streamlit
- FastAPI
- Microsoft RAD-DINO
- PyTorch
- Hugging Face Transformers
- Pillow
- ReportLab

## ⚙️ Project Workflow

1. Upload an X-ray image.
2. The frontend sends the image to the FastAPI backend.
3. RAD-DINO extracts visual features from the image.
4. The API returns feature information.
5. Streamlit displays the results.
6. The user can download a PDF report.

## ▶️ How to Run

### 1. Start the FastAPI backend

Open a terminal in the project folder and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

### 2. Start the Streamlit frontend

Open a second terminal in the same project folder and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 3. Open the application

Open the Local URL shown in the terminal, usually:

http://localhost:8501

## 📊 Example Output

- Image dimensions: 716 × 376 pixels
- Feature tokens: 1370
- Feature dimensions: 768
- Processing time: Varies depending on the system

These are example results from one test image.

## ⚠️ Disclaimer

MedLens AI is an academic prototype for medical image feature extraction. It does not diagnose diseases or provide clinically validated medical reports. It must not be used for patient-care decisions.

## 👩‍💻 Project Status

Prototype development in progress.
