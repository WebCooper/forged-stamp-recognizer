from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src.config import MODELS_DIR
from src.inference import run_full_pipeline_on_bgr


st.set_page_config(
    page_title="Forged Stamp Recognizer",
    page_icon="🖋️",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def image_bytes_to_bgr(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not read the uploaded image")
    return image_bgr


st.title("Forged Stamp Recognizer")
st.write(
    "Upload a scanned document image and the app will detect the stamp region, classify it, and show the result."
)

with st.sidebar:
    st.header("Demo settings")
    model_path = st.text_input(
        "Model path",
        value=str(MODELS_DIR / "stamp_resnet50_final.keras"),
    )
    display_width = st.slider("Processing width", min_value=600, max_value=1400, value=900, step=100)
    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)

uploaded_file = st.file_uploader("Choose a document image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    try:
        image_bgr = image_bytes_to_bgr(uploaded_file)
        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"Model not found: {model_file}")
            st.stop()

        with st.spinner("Running stamp detection..."):
            model = load_model(str(model_file))
            result = run_full_pipeline_on_bgr(
                image_bgr=image_bgr,
                model=model,
                display_width=display_width,
                decision_threshold=threshold,
            )
            if result["status"] != "ok":
                st.error(result["message"])
                st.stop()

            roi_bgr = result["roi_bgr"]
            roi_info = result["roi_info"]
            annotated = result["annotated_bgr"]
            raw_prob = float(result["raw_prob"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded image")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected stamp ROI")
            st.image(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Prediction")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Verdict", result["class"])
        metric_col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        metric_col3.metric("Raw forged probability", f"{raw_prob * 100:.1f}%")

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated document", use_container_width=True)

        with st.expander("Technical details"):
            st.write({
                "Bounding box": roi_info["bbox_original"],
                "Threshold": threshold,
                "Model": str(model_file),
                "Raw forged probability": raw_prob,
                "Inference ms": result["inference_ms"],
                "Total ms": result["total_ms"],
            })

    except Exception as exc:
        st.exception(exc)
else:
    st.info("Upload a stamp document image to start the demo.")