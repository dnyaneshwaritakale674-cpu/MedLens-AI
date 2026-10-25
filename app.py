

import streamlit as st
import requests
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MedLens AI | Medical Imaging",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f8fc 0%, #edf5f7 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #102a43, #155e75);
    color: white;
    margin-bottom: 25px;
}
.hero h1 { color: white; font-size: 38px; margin-bottom: 8px; }
.hero p { color: #d8edf5; font-size: 16px; }
.eyebrow {
    color: #5eead4;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 700;
}
.panel {
    background: white;
    border: 1px solid #dce6ee;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}
.small-note {
    color: #526579;
    font-size: 13px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🩺 MedLens AI")
    st.caption("Medical Imaging Intelligence")
    st.divider()
    st.markdown("### Project")
    st.write("**Model:** Microsoft RAD-DINO")
    st.write("**Backend:** FastAPI")
    st.write("**Frontend:** Streamlit")
    st.write("**Task:** X-ray feature extraction")
    st.divider()

    try:
        health = requests.get(f"{API_URL}/", timeout=3)
        if health.ok:
            st.success("API Connected")
        else:
            st.warning("API returned an error")
    except requests.RequestException:
        st.error("API Offline")
        st.caption("Start your FastAPI server.")

    st.divider()
    st.caption("Academic demonstration only")

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-POWERED MEDICAL IMAGING</div>
    <h1>MedLens AI</h1>
    <p>Explore medical image features using a pretrained
    vision model and a simple, connected AI pipeline.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📤 Upload an X-ray")
    st.write("Choose a PNG or JPEG image to begin.")
    uploaded_file = st.file_uploader(
        "Select image",
        type=["png", "jpg", "jpeg"],
        help="Use a public, de-identified sample image for this demo."
    )

    if uploaded_file:
        st.image(
            uploaded_file.getvalue(),
            caption=uploaded_file.name,
            use_container_width=True
        )
        st.caption(
            f"File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB"
        )

    analyze = st.button(
        "✨ Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <h4>🔬 How it works</h4>
        <p>01 &nbsp; Upload an image</p>
        <p>02 &nbsp; FastAPI validates the file</p>
        <p>03 &nbsp; RAD-DINO extracts visual features</p>
        <p>04 &nbsp; Results appear in this dashboard</p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Dashboard")
    st.write("Your processing results will appear here.")

    if analyze and uploaded_file:
        try:
            with st.spinner("RAD-DINO is processing the image..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
                response = requests.post(
                    f"{API_URL}/analyze",
                    files=files,
                    timeout=180
                )

            if response.ok:
                result = response.json()
                st.session_state["medlens_result"] = result
                st.session_state["medlens_error"] = None
            else:
                st.session_state["medlens_result"] = None
                st.session_state["medlens_error"] = (
                    f"API error {response.status_code}: {response.text}"
                )

        except requests.RequestException as error:
            st.session_state["medlens_result"] = None
            st.session_state["medlens_error"] = str(error)

    
    result = st.session_state.get("medlens_result")
    error = st.session_state.get("medlens_error")

    if error:
        st.error("Processing failed")
        st.write(error)

    elif result:
        st.success("Feature extraction completed")

        size = result.get("image_size", {})
        shape = result.get("feature_tensor_shape", [])

        a, b, c = st.columns(3)

        a.metric(
            "Feature tokens",
            result.get("feature_count", "—")
        )
        b.metric(
            "Feature dimensions",
            result.get("feature_dimension", "—")
        )
        c.metric(
            "Processing time",
            f"{result.get('processing_time_seconds', '—')} sec"
        )

        st.markdown("**Image dimensions**")
        st.write(
            f"{size.get('width', '—')} × "
            f"{size.get('height', '—')} pixels"
        )

        st.markdown("**Feature tensor shape**")
        st.code(str(shape), language="python")

        st.markdown("**Processing status**")
        st.write(result.get("analysis_status", "Unknown"))

        
        with st.expander("View complete API response"):
            st.json(result)

       
        

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()

        story = []
        story.append(Paragraph("MedLens AI - Analysis Report", styles["Title"]))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Research prototype: Image feature extraction only.", styles["Normal"]))
        story.append(Spacer(1, 12))

        for key, value in result.items():
            story.append(Paragraph(f"<b>{key}</b>: {str(value)}", styles["Normal"]))
            story.append(Spacer(1, 8))

        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "Medical diagnosis is not provided. This report is not for clinical use.",
            styles["Italic"]
        ))

        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="MedLens_Analysis_Report.pdf",
            mime="application/pdf"
        )



    else:
        st.info("Upload an image and select Analyze Image to see results.")
        st.markdown("**Waiting for image**")
        st.progress(0)
        st.caption("No image has been analyzed in this session.")

    st.markdown('</div>', unsafe_allow_html=True)


st.divider()
st.caption(
    "MedLens AI • Student ML/DL project • "
    "RAD-DINO feature extraction demo"
)
st.warning(
    "Safety notice: This prototype extracts image features only. "
    "It does not diagnose disease or generate a clinically validated "
    "medical report. Do not use it for patient-care decisions."
)
