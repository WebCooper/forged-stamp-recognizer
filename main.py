# =====================================================================
# FORGED STAMP RECOGNIZER - MASTER PIPELINE SCRIPT
# This is the main orchestrator for our project.
# Basically, it runs all the steps in order so we don't have to manually 
# trigger each script one by one.
# =====================================================================

import sys
from pathlib import Path

# Importing the modular pipeline stages we wrote in the src folder
from src.extract import run_extraction
from src.cleaner import run_cleaner
from src.dataset import generate_balanced_dataset
from src.train import train_model
from src.evaluate import evaluate_model


def run_pipeline():
    print("=" * 50)
    print("   FORGED STAMP RECOGNIZER - MASTER PIPELINE   ")
    print("=" * 50)

    # ---------------------------------------------------------
    # STEP 1: Extract Raw Stamps
    # ---------------------------------------------------------
    # First we crop the stamp regions of interest (ROIs) from the scan images.
    # This uses the image segmentation and geometric contour scoring we built.
    print("\n[STEP 1/5] Extracting Raw ROIs from Scans...")
    run_extraction()

    # ---------------------------------------------------------
    # STEP 2: Human-in-the-Loop Cleaning
    # ---------------------------------------------------------
    # The automatic cropping is mostly good, but sometimes it crops random round logos.
    # So we built this GUI cleaner tool to quickly filter out the bad crops.
    print("\n[STEP 2/5] Launching Manual Cleaning Tool...")
    print("A window will open. Press Y to keep, N to delete.")
    run_cleaner()

    # Safety pause: If you quit the GUI early (pressed 'Q'), we shouldn't proceed
    # to training with incomplete/uncleaned data.
    proceed = input(
        "\nCleaning finished. Proceed to Data Augmentation and Training? (y/n): "
    )
    if proceed.lower() != "y":
        print("Pipeline aborted by user. You can resume later.")
        sys.exit()

    # ---------------------------------------------------------
    # STEP 3: Balance and Augment
    # ---------------------------------------------------------
    # The classes might have different number of images after cleaning.
    # Deep learning models get biased if classes are unbalanced, so we do data augmentation
    # to hit exactly TARGET_CLASS_SIZE for both genuine and forged classes.
    print("\n[STEP 3/5] Generating Balanced & Augmented Dataset...")
    generate_balanced_dataset()

    # ---------------------------------------------------------
    # STEP 4: Train the ResNet50 Model
    # ---------------------------------------------------------
    # We are using transfer learning with ResNet50
    # because it converges way faster than our custom CNN models.
    print("\n[STEP 4/5] Training the Deep Learning Classifier...")
    train_model()

    # ---------------------------------------------------------
    # STEP 5: Evaluate Performance
    # ---------------------------------------------------------
    # how our model actually does on the validation set.
    # Prints out precision, recall, f1-score, and the confusion matrix.
    print("\n[STEP 5/5] Evaluating Model Performance...")
    evaluate_model()

    print("\n" + "=" * 50)
    print("              PIPELINE COMPLETE              ")
    print("=" * 50)


if __name__ == "__main__":
    # Standard Python entrypoint check. Run the pipeline if this script is executed directly.
    run_pipeline()

