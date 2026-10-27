

import streamlit as st
import requests
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def create_pdf_report(item):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 60

    pdf.setTitle("MedLens AI Technical Report")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, y, "MedLens AI")
    y -= 30

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Technical Processing Report")
    y -= 35

    pdf.setFont("Helvetica", 11)

    details = [
        f"Image filename: {item['filename']}",
        f"AI Model: {item['model']}",
        f"Processing time: {item['processing_time']} seconds",
        f"Feature vector length: {item['feature_length']}",
        f"Feature shape: {item.get('feature_shape', 'N/A')}",
        "",
        "Purpose: Research and technical demonstration.",
        "",
        "Safety notice:",
        "This report contains technical processing information only.",
        "RAD-DINO feature extraction does not provide a diagnosis.",
        "Do not use these results for medical decisions."
    ]

    for line in details:
        pdf.drawString(50, y, line)
        y -= 22

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()

st.set_page_config(
    page_title="MedLens AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

API_URL = "http://127.0.0.1:8000/analyze"

# Professional styling
st.markdown("""
<style>
    .stApp {
        background-color: #f4f8fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        background: linear-gradient(120deg, #063b5c, #087e8b);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 5px;
    }
    .hero p {
        color: #e4f5fa;
        font-size: 17px;
    }
    .section-title {
        color: #12344d;
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce8ef;
        padding: 18px;
        border-radius: 14px;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        padding: 15px;
        border: 1px dashed #8bb8c8;
        border-radius: 14px;
    }
    .notice {
        background: #e6f5f4;
        padding: 14px;
        border-radius: 10px;
        color: #164e52;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🩺 MedLens AI")
    st.caption("Medical Imaging Research Dashboard")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 **Image Analysis**")
    st.markdown("🕘 **Analysis History**")

    st.divider()
    st.markdown("### AI Model")
    st.code("microsoft/rad-dino")
    st.caption("Research feature extraction only")

# Header
st.markdown("""
<div class="hero">
    <h1>🩺 MedLens AI</h1>
    <p>Medical image intelligence for research and exploration.</p>
    <p>Upload a sample X-ray and extract image features with RAD-DINO.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📤 Upload Medical Image</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload a sample PNG or JPEG image for this research demo."
)

result = None

if uploaded_file is not None:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### 📷 Image Preview")
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with right:
        st.markdown("#### 📋 Image Information")
        st.write("**File name:**", uploaded_file.name)
        st.write("**File size:**", f"{uploaded_file.size / 1024:.2f} KB")
        st.write("**File type:**", uploaded_file.type)

        st.markdown(
            '<div class="notice">Ready for research feature extraction.</div>',
            unsafe_allow_html=True
        )

    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Loading model and extracting image features..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=300
                )

                if response.ok:
                    result = response.json()

                    st.session_state.analysis_history.append({
                        "filename": result.get("filename", uploaded_file.name),
                        "model": result.get("model", "N/A"),
                        "processing_time": result.get("processing_time_seconds", 0),
                        "feature_length": result.get("feature_vector_length", "N/A"),
                        "feature_shape": result.get("feature_shape", [])
                    })

                    st.success("Image feature extraction completed!")
                else:
                    st.error(
                        f"Backend error ({response.status_code}): {response.text}"
                    )

            except requests.Timeout:
                st.error("Analysis took too long. Please try again.")

            except requests.RequestException:
                st.error(
                    "Cannot connect to the backend. Make sure FastAPI "
                    "is running on port 8000."
                )

# Display latest successful result
if st.session_state.analysis_history:
    latest = st.session_state.analysis_history[-1]

    st.divider()
    st.markdown('<div class="section-title">📊 Latest Analysis Results</div>',
                unsafe_allow_html=True)

    st.info(
        "This is a technical processing summary, not a clinical report "
        "or medical diagnosis."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Processing Time", f"{latest['processing_time']} sec")

    with c2:
        st.metric("Feature Vector Length", latest["feature_length"])

    with c3:
        st.metric("Model Status", "Successful")

    st.write("**AI Model:**", latest["model"])
    st.write("**Feature Shape:**", latest["feature_shape"])

    st.caption(
        "Extracted numerical features are not disease probabilities "
        "and should not be used to make medical decisions."
    )

# Download PDF report
if st.session_state.analysis_history:
    st.subheader("📄 Download Technical Report")

    latest_report = st.session_state.analysis_history[-1]

    pdf_data = create_pdf_report(latest_report)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_data,
        file_name="MedLens_AI_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# History
st.divider()
st.markdown('<div class="section-title">🕘 Analysis History</div>',
            unsafe_allow_html=True)

if st.session_state.analysis_history:
    for i, item in enumerate(
        reversed(st.session_state.analysis_history), start=1
    ):
        with st.expander(f"{i}. {item['filename']}"):
            st.write("**AI Model:**", item["model"])
            st.write("**Processing Time:**", f"{item['processing_time']} sec")
            st.write("**Feature Vector Length:**", item["feature_length"])
            st.write("**Feature Shape:**", item["feature_shape"])
else:
    st.info("No analysis history yet. Upload an image and click Analyze Image.")

st.divider()

st.caption(
    "MedLens AI | Research demonstration only | "
    "Not intended for medical diagnosis or treatment decisions."
)
