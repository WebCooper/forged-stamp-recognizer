from pathlib import Path

# Dataset root
DATASET_ROOT = Path(r"H:\UOR\7th sem\Computer Vision\final_dataset")

# Class mapping
CLASS_MAPPING = {
    "class_0_genuine": 0,
    "class_1_forged": 1
}

LABEL_NAMES = {
    0: "genuine",
    1: "forged"
}

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEGMENTATION_DEBUG_DIR = OUTPUTS_DIR / "segmentation_debug"
ROI_CROPS_DIR = OUTPUTS_DIR / "roi_crops"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create output folders if missing
for folder in [
    OUTPUTS_DIR,
    SEGMENTATION_DEBUG_DIR,
    ROI_CROPS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)