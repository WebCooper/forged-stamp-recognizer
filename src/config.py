# =====================================================================
# GLOBAL CONFIGURATION VARIABLES
# We put all our paths, image sizes, and neural network parameters here
# so they are centralized. If we want to change epochs or batch size,
# we only change them here instead of searching through 10 files. Sweet!
# =====================================================================

from pathlib import Path

# --- BASE PATHS ---
# Path(__file__).resolve().parent.parent gets the absolute path of the root directory.
# This prevents pathing errors if we run scripts from notebooks/ or from the root folder.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "final_dataset"

# --- PIPELINE OUTPUT DIRECTORIES ---
RAW_ROI_DIR = BASE_DIR / "outputs" / "raw_rois"        # Step 1: Saves raw crops
BALANCED_ROI_DIR = BASE_DIR / "outputs" / "balanced_rois"  # Step 3: Saves augmented training data
MODEL_OUTPUT_DIR = BASE_DIR / "outputs" / "models"       # Step 4: Holds trained .keras model files

# Ensure all directories exist automatically
# If the folders don't exist, mkdir(parents=True) makes them on the fly.
# No more manual folder creation errors when running on a teammate's laptop!
for folder in ["genuine", "forged"]:
    (RAW_ROI_DIR / folder).mkdir(parents=True, exist_ok=True)
    (BALANCED_ROI_DIR / folder).mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- COMPUTER VISION HYPERPARAMETERS ---
DISPLAY_WIDTH = 900  # We resize documents to this width for processing/display
ROI_SIZE = 224       # Crop size for localizing the stamps (224x224)

# --- DEEP LEARNING HYPERPARAMETERS ---
IMG_SIZE = (224, 224)     # ResNet50 expects 224x224 image dimensions
BATCH_SIZE = 16          # Kept at 16 so our local laptops don't run out of GPU memory
SEED = 42                # Seed 42 for reproducibility of validation splits (and hitchhiker reference)
EPOCHS_BASE = 15         # Number of epochs to train just the final dense classifier layer
EPOCHS_FINETUNE = 10     # Number of epochs for fine-tuning the top layers of ResNet50
TARGET_CLASS_SIZE = 1200 # Target dataset size per class after data augmentation

