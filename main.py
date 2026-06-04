import sys
from pathlib import Path

# Import the pipeline stages from your src modules
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
    print("\n[STEP 1/5] Extracting Raw ROIs from Scans...")
    run_extraction()

    # ---------------------------------------------------------
    # STEP 2: Human-in-the-Loop Cleaning
    # ---------------------------------------------------------
    print("\n[STEP 2/5] Launching Manual Cleaning Tool...")
    print("A window will open. Press Y to keep, N to delete.")
    run_cleaner()

    # Add a safety pause in case you pressed 'Q' to quit cleaning early
    proceed = input(
        "\nCleaning finished. Proceed to Data Augmentation and Training? (y/n): "
    )
    if proceed.lower() != "y":
        print("Pipeline aborted by user. You can resume later.")
        sys.exit()

    # ---------------------------------------------------------
    # STEP 3: Balance and Augment
    # ---------------------------------------------------------
    print("\n[STEP 3/5] Generating Balanced & Augmented Dataset...")
    generate_balanced_dataset()

    # ---------------------------------------------------------
    # STEP 4: Train the ResNet50 Model
    # ---------------------------------------------------------
    print("\n[STEP 4/5] Training the Deep Learning Classifier...")
    train_model()

    # ---------------------------------------------------------
    # STEP 5: Evaluate Performance
    # ---------------------------------------------------------
    print("\n[STEP 5/5] Evaluating Model Performance...")
    evaluate_model()

    print("\n" + "=" * 50)
    print("              PIPELINE COMPLETE              ")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
