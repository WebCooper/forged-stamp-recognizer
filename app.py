# =====================================================================
# STREAMLIT USER INTERFACE FOR FORENSIC DOCUMENT VERIFICATION
# This is our frontend app built with Streamlit. We wanted to make a clean,
# interactive dashboard so anyone can test our model without running python scripts.
# =====================================================================

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from pathlib import Path

# Importing helper variables and functions we defined in src/
from src.config import MODEL_OUTPUT_DIR, IMG_SIZE
from src.segmentation import segment_stamp, preprocess_image
from src.localization import extract_roi

# --- PAGE CONFIGURATION ---
# Sets up the browser tab title, icon, and switches layout to wide mode
st.set_page_config(page_title="Document Stamp Verifier", page_icon="🔎", layout="wide")


# --- CACHE THE MODEL ---
# This is a super handy Streamlit decorator! Without it, Streamlit would reload
# the entire neural network weights from the disk every single time the user interacts
# with a slider or uploads a new file. That would cause massive lag lol.
@st.cache_resource
def load_classifier():
    model_path = MODEL_OUTPUT_DIR / "final_stamp_classifier.keras"
    return tf.keras.models.load_model(model_path)


# Load the model once into memory
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
    # 1. Read the uploaded file bytes and decode them into OpenCV BGR format.
    # We convert raw bytes to numpy array, then use cv2.imdecode. Classic trick!
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # type: ignore # Convert BGR to RGB for Streamlit plotting

    # Set up 3 columns side-by-side to show the full pipeline visualization
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Original Document")
        st.image(image_rgb, use_container_width=True)

    with st.spinner("Running Computer Vision Pipeline..."):
        try:
            # 2. Run the custom computer vision pipeline
            # Get height and width to calculate aspect ratio scaling
            h, w = image_bgr.shape[:2] # type: ignore
            scale = 900 / w
            resized_bgr = cv2.resize(image_bgr, (900, int(h * scale))) # type: ignore

            # Apply segmentation mask and extract target cropped Region of Interest (ROI)
            mask = segment_stamp(resized_bgr)
            roi = extract_roi(resized_bgr, mask)

            if roi is None:
                # If CV Hough Circle / Contour scores fail, show a friendly warning.
                st.error(
                    "Failed to detect a stamp in this image. Try another document."
                )
            else:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("2. Extracted ROI")
                    st.image(
                        roi_rgb,
                        use_container_width=True,
                        caption="Hough Circle / Contour Crop",
                    )

                # 3. Model Inference (ResNet50 classification)
                with col3:
                    st.subheader("3. Forensic Analysis")

                    # Resize the cropped ROI to 224x224 because that's what ResNet50 expects
                    input_img = cv2.resize(roi_rgb, IMG_SIZE)
                    
                    # Expand dimensions to (1, 224, 224, 3) because Keras model.predict
                    # expects a batch of images, even if batch size is just 1.
                    input_array = np.expand_dims(input_img, axis=0)

                    # Time the prediction just to show the user how fast it is
                    start_time = time.time()
                    prediction_prob = model.predict(input_array, verbose=0)[0][0]
                    inference_time = time.time() - start_time

                    # NOTE: Class mapping is alphabetical:
                    # 'forged' is Class 0 (values from 0.0 to 0.49)
                    # 'genuine' is Class 1 (values from 0.50 to 1.0)
                    is_genuine = prediction_prob >= 0.5

                    if is_genuine:
                        st.success(f"✅ **GENUINE STAMP VERIFIED**")
                        # Closer to 1.0 means higher confidence of being genuine
                        confidence = prediction_prob * 100
                    else:
                        st.error(f"🚨 **FORGED STAMP DETECTED**")
                        # Closer to 0.0 means higher confidence of being forged
                        confidence = (1.0 - prediction_prob) * 100

                    # Render metrics on screen
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                    st.caption(f"Inference Speed: {inference_time:.3f} seconds")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

