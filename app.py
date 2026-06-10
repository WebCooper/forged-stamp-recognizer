import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from pathlib import Path

# Import your custom pipeline modules
from src.config import MODEL_OUTPUT_DIR, IMG_SIZE
from src.segmentation import segment_stamp, preprocess_image
from src.localization import extract_roi

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Document Stamp Verifier", page_icon="🔎", layout="wide")


# --- CACHE THE MODEL ---
# This ensures the model only loads once, keeping the web app super fast!
@st.cache_resource
def load_classifier():
    model_path = MODEL_OUTPUT_DIR / "final_stamp_classifier.keras"
    return tf.keras.models.load_model(model_path)


model = load_classifier()

# --- UI HEADER ---
st.title("🔎 Image-Based Document Stamp Verification System")
st.markdown("""
Upload a scanned document containing an organizational stamp. The AI pipeline will automatically 
locate the stamp, extract it, and analyze the microscopic texture to determine if it is a 
**Genuine (Wet-Ink)** impression or a **Forged (Digitally Printed)** copy.
""")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload Scanned Document (PNG, JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # 1. Read the image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # type: ignore # For Streamlit display

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Original Document")
        st.image(image_rgb, use_container_width=True)

    with st.spinner("Running Computer Vision Pipeline..."):
        try:
            h, w = image_bgr.shape[:2]
            scale = 900 / w
            resized_bgr = cv2.resize(image_bgr, (900, int(h * scale)))

            mask = segment_stamp(resized_bgr)
            roi = extract_roi(resized_bgr, mask)

            if roi is None:
                st.error(
                    "Failed to detect a stamp in this image. Try another document."
                )
            else:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # Prepare image for ResNet50 FIRST so we can display it
                input_img = cv2.resize(roi_rgb, IMG_SIZE)

                with col2:
                    st.subheader("2. Extracted ROI")
                    # Set use_container_width=False to prevent stretching/blurring
                    st.image(
                        input_img,
                        use_container_width=False,
                        caption=f"Model Feed {IMG_SIZE}",
                    )

                # 3. Model Inference
                with col3:
                    st.subheader("3. Forensic Analysis")

                    # Prepare image for ResNet50
                    input_img = cv2.resize(roi_rgb, IMG_SIZE)
                    input_array = np.expand_dims(input_img, axis=0)

                    start_time = time.time()
                    prediction_prob = model.predict(input_array, verbose=0)[0][0]
                    inference_time = time.time() - start_time

                    # NOTE: Alphabetical loading mapping:
                    # 0.0 to 0.49 = forged (Class 0)
                    # 0.50 to 1.0 = genuine (Class 1)

                    is_genuine = prediction_prob >= 0.5

                    if is_genuine:
                        st.success(f"✅ **GENUINE STAMP VERIFIED**")
                        # If prediction is 0.95, it is 95% confident it's genuine
                        confidence = prediction_prob * 100
                    else:
                        st.error(f"🚨 **FORGED STAMP DETECTED**")
                        # If prediction is 0.05, it is 95% confident it's forged
                        confidence = (1.0 - prediction_prob) * 100

                    st.metric("Confidence Score", f"{confidence:.2f}%")
                    st.caption(f"Inference Speed: {inference_time:.3f} seconds")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
