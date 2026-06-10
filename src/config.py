from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "final_dataset"

# New Pipeline Folders
RAW_ROI_DIR = BASE_DIR / "outputs" / "raw_rois"  # Step 1: Holds initial crops
BALANCED_ROI_DIR = (
    BASE_DIR / "outputs" / "balanced_rois"
)  # Step 3: Holds final training data
MODEL_OUTPUT_DIR = BASE_DIR / "outputs" / "models"

# Ensure directories exist
for folder in ["genuine", "forged"]:
    (RAW_ROI_DIR / folder).mkdir(parents=True, exist_ok=True)
    (BALANCED_ROI_DIR / folder).mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Computer Vision Hyperparameters
DISPLAY_WIDTH = 900
ROI_SIZE = 224

# Deep Learning Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS_BASE = 15
EPOCHS_FINETUNE = 10
TARGET_CLASS_SIZE = 1200  # For dataset balancing
