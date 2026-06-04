# Image-Based Document Stamp Verification System
### EC7205 — Image Processing and Computer Vision
**Department of Electrical and Information Engineering, University of Ruhuna**

---

| | |
|---|---|
| **S.A.U. Fernando** | EG/2021/4511 |
| **J.S. Karunarathna** | EG/2021/4599 |
| **A.D.H. Karunathilake** | EG/2021/4604 |
| **M.M.P.L. Manthilaka** | EG/2021/4671 |

---
## 1. Problem Statement

Organisational seals and stamps are used across legal, academic, and administrative institutions as a primary means of authenticating documents. However, the widespread availability of high-quality flatbed scanners and colour printers has made stamp forgery increasingly accessible. A forged stamp can be produced simply by scanning a genuine stamped document and reprinting it — a technique known as the **Scan-Print-Scan (SPS)** method.

Existing stamp verification approaches fall into two categories, each with limitations:

- **Classical image processing methods** (HSV thresholding, morphological filtering) are fast and interpretable but fail in the presence of noise, overlapping text, and varying scan conditions.
- **Deep learning models** achieve high accuracy but require large, labelled datasets that are not publicly available for document forensics.

Neither category alone addresses the core forensic challenge: distinguishing the **microscopic texture of genuine wet-ink** from the **halftone dot patterns of a digitally printed reproduction**.

## 2. Proposed Solution

This project develops a **hybrid computer vision pipeline** that combines the strengths of both approaches:

1. **Classical image processing** handles the complex, noisy extraction problem — isolating the stamp from overlapping text, signatures, and background using HSV colour segmentation and morphological operations.

2. **Transfer learning with ResNet50** handles the forensic classification problem — analysing the extracted stamp region for microscopic texture differences between genuine wet-ink impressions and digital forgeries.

The system classifies a document stamp as one of two classes:

| Class | Label | Description |
|---|---|---|
| Genuine | 0 | Real wet-ink stamp impression, physically applied with a rubber stamp and ink pad |
| Forged | 1 | Digital reproduction created via the SPS method using inkjet or laser printing |


## 3. System Pipeline

The full pipeline runs across six stages, each implemented in a separate notebook:

```
Raw scanned document
        │
        ▼
┌─────────────────────────────┐
│  HSV Colour Segmentation    │  Isolate blue/violet stamp ink pixels
│  Morphological Filtering    │  Fill gaps, remove text noise
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Contour Analysis           │  Score regions by circularity & compactness
│  ROI Extraction             │  Crop stamp region at full resolution
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  ResNet50 Transfer Learning │  Classify ink texture: genuine vs forged
└─────────────────────────────┘
        │
        ▼
  Classification result
  (Genuine / Forged)
```

## 4. Dataset

A custom forensic dataset of approximately **300 scanned document images** was constructed because no suitable public dataset exists for this task.

**Genuine samples (Class 0)**
- Three unique rubber stamps applied with a violet/blue ink pad
- Stamped under varying pressure, orientation, and lighting conditions
- Scanned at **300 DPI** and **600 DPI** using a flatbed scanner

**Forged samples (Class 1)**
- Genuine stamped documents digitally reprinted using:
  - A **colour laser printer** (introduces toner artifacts and edge aliasing)
  - An **inkjet printer** (introduces halftone dot patterns)
- Reprinted documents re-scanned under identical conditions to genuine samples (SPS method)

**Data augmentation**
To improve generalisation, augmentation is applied during training: random rotation (±29°), zoom (±8%), translation (±5%), and contrast variation (±15%).

**Dataset structure on disk:**
```
final_dataset/
├── class_0_genuine/
│   └── *.png
└── class_1_forged/
    └── *.png
```
## 5. Notebook Structure

Run the notebooks **in order**. Each notebook builds on the outputs of the previous one.

| Notebook | Purpose | Input | Output |
|---|---|---|---|
| `00_environment_check.ipynb` | Verify libraries and dataset path | — | Console output |
| `01_dataset_audit.ipynb` | Audit raw dataset for class balance, resolution, readability | `final_dataset/` | `outputs/reports/dataset_metadata.csv` |
| `02_preprocessing_segmentation.ipynb` | Develop and visualise HSV + morphology pipeline on one image | One sample image | Visual inspection |
| `03_stamp_segmentation_contours.ipynb` | Develop contour-based stamp detection (used in final pipeline) | One sample image | `find_stamp_candidates_by_contours` function |
| `03_geometric_localization.ipynb` | Alternative Hough Circle Transform approach (exploratory) | One sample image | Visual inspection |
| `04_generate_roi_dataset.ipynb` | Run detection pipeline across full dataset, save 224×224 ROIs | `final_dataset/` | `outputs/roi_dataset_v3/` |
| `05_train_classifier.ipynb` | Train ResNet50 classifier, evaluate, save model | `outputs/roi_dataset_v3/` | `outputs/models/stamp_resnet50_final.keras` |

> **Note:** `03_geometric_localization.ipynb` is an exploratory notebook documenting the Hough Circle Transform approach that was tested before settling on the contour method. It does not need to be run to produce the final outputs.

## 6. Environment Setup

**Requirements**

```
Python        3.11.x
OpenCV        4.x        (pip install opencv-python)
TensorFlow    2.x        (pip install tensorflow)
scikit-learn             (pip install scikit-learn)
pandas                   (pip install pandas)
matplotlib               (pip install matplotlib)
tqdm                     (pip install tqdm)
```

**Recommended setup (Windows)**

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install opencv-python tensorflow scikit-learn pandas matplotlib tqdm

# Launch Jupyter
jupyter notebook
```

**Before running any notebook**, update the `DATASET_ROOT` / `IMAGE_PATH` variable at the top of each notebook to point to your local copy of `final_dataset/`.

## 7. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Colour space for segmentation | HSV | Hue channel separates coloured ink from black text without training data |
| Stamp localisation method | Contour scoring (circularity + compactness) | More robust than Hough Circle Transform on partially occluded or irregular stamps |
| ROI extraction resolution | Full original resolution | Preserves microscopic ink texture for classifier; resized only after cropping |
| Classifier backbone | ResNet50 | Strong ImageNet features + residual connections handle texture well; manageable on CPU |
| Training strategy | Two-phase transfer learning | Phase 1 adapts the head; Phase 2 fine-tunes top layers without destroying ImageNet features |
| Output size | 224 × 224 px | Matches ResNet50 input; balances texture detail and memory |

---
*Image-Based Document Stamp Verification System — EC7205, University of Ruhuna, 2026*
