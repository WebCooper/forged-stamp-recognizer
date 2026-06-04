
# Image-Based Document Stamp Verification System

An automated, hybrid computer vision and deep learning pipeline designed to verify the authenticity of organizational stamps. This system distinguishes genuine wet-ink impressions from digitally printed forgeries (Scan-Print-Scan method) using morphological processing, geometric localization, and transfer learning-based texture analysis.

## Key Features
* **Robust Stamp Extraction:** Utilizes HSV color space segmentation, morphological filtering, and Hough Circle Transforms to precisely crop circular stamps from noisy document backgrounds.
* **Human-in-the-Loop Validation:** Includes a custom OpenCV interactive UI to quickly clean and verify extracted Regions of Interest (ROIs).
* **Targeted Augmentation:** Automatically balances class disparities by dynamically generating augmented variations (rotation, Gaussian noise, contrast shifting).
* **Deep Learning Classifier:** Fine-tuned ResNet50 model that learns microscopic textures to differentiate continuous wet-ink diffusion from periodic halftone digital prints.
* **Interactive Web App:** A sleek Streamlit dashboard for real-time inference and presentation.

---

## Project Architecture

```text
forged-stamp-recognizer/
│
├── final_dataset/                 # Raw scanned documents (Genuine & Forged)
├── outputs/                       
│   ├── raw_rois/                  # Initial stamp crops before cleaning
│   ├── balanced_rois/             # Cleaned, augmented, and balanced dataset
│   └── models/                    # Saved ResNet50 .keras weights
│
├── notebooks/                     # Exploratory Data Analysis & scratchpads
│
├── src/                           # Core Pipeline Modules
│   ├── config.py                  # Centralized hyperparameters and paths
│   ├── extract.py                 # Computer vision extraction logic
│   ├── cleaner.py                 # Interactive OpenCV data cleaning UI
│   ├── dataset.py                 # Dynamic dataset balancing & augmentation
│   ├── train.py                   # 2-step Transfer Learning (ResNet50)
│   └── evaluate.py                # Metrics generation (F1-score, Confusion Matrix)
│
├── main.py                        # Master script to run the pipeline sequentially
├── app.py                         # Streamlit web application for live demos
├── requirements.txt               # Python dependencies
└── .gitignore                     # Version control rules

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone git@github.com:WebCooper/forged-stamp-recognizer.git
cd forged-stamp-recognizer
```


2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```


3. **Install the dependencies:**
```bash
pip install -r requirements.txt
```


*(Ensure `opencv-python`, `tensorflow`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `matplotlib` are included in your requirements.txt)*
4. **Prepare the Data:**
Place your raw scanned document images into the `final_dataset` folder, organized by class:
* `final_dataset/class_0_genuine/`
* `final_dataset/class_1_forged/`



---

## Usage: Running the Pipeline

You can run the entire extraction, cleaning, and training pipeline using the master control script:

```bash
python main.py
```

**Pipeline Stages:**

1. **Extraction:** Scans raw documents and mathematically isolates the stamps.
2. **Cleaning:** An interactive window will appear. Press **[Y]** to keep valid stamps or **[N]** to delete bad crops (e.g., signatures or logos). Press **[Q]** to save and quit.
3. **Augmentation:** The system will automatically calculate the required augmentations to create a perfectly 50/50 balanced dataset based on your surviving clean stamps.
4. **Training:** Fine-tunes the ResNet50 classifier.
5. **Evaluation:** Outputs the final metrics.

---

## Running the Demo App

To launch the interactive presentation dashboard, run:

```bash
streamlit run app.py
```

This will open a local web server where you can upload a new scanned document and watch the AI locate, extract, and verify the stamp in real-time.

---

## Model Performance

The hybrid pipeline achieves state-of-the-art performance on our custom forensic dataset:

* **Accuracy:** 99.0%
* **F1-Score:** 0.99
* **Inference Speed:** ~0.04 seconds per Region of Interest

---

## Technologies Used

* **Computer Vision:** OpenCV (cv2)
* **Deep Learning:** TensorFlow / Keras (ResNet50)
* **Data Manipulation:** NumPy, Pandas
* **Evaluation:** Scikit-Learn, Matplotlib
* **Web UI:** Streamlit

---

*Developed for the  EC7205 Image Processing and Computer Vision Module - Faculty of Engineering, University of Ruhuna.*
