from pathlib import Path

from dotenv import load_dotenv


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
ROI_DATASET_DIR = OUTPUTS_DIR / "roi_dataset_v3"
SEGMENTATION_DEBUG_DIR = OUTPUTS_DIR / "segmentation_debug"
ROI_CROPS_DIR = OUTPUTS_DIR / "roi_crops"


load_dotenv(PROJECT_ROOT / ".env")


def get_dataset_root(default: str | Path | None = None) -> Path:
    """Return the dataset root from the environment or a supplied default."""
    import os

    candidate = os.getenv("STAMP_DATASET_ROOT") or default
    if candidate is None:
        return PROJECT_ROOT / "data" / "final_dataset"
    return Path(candidate)


# Class mapping
CLASS_MAPPING = {
    "class_0_genuine": 0,
    "class_1_forged": 1,
}

LABEL_NAMES = {
    0: "genuine",
    1: "forged",
}


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
