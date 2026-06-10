# =====================================================================
# MODEL EVALUATION SCRIPT
# This script computes standard classification metrics (Accuracy, Precision, 
# Recall, F1-Score) and the Confusion Matrix on the validation dataset.
# Helps us verify how well the ResNet50 model generalizes to unseen stamp images.
# =====================================================================

import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.config import MODEL_OUTPUT_DIR
from src.train import build_dataset


def evaluate_model():
    # Load the validation dataset (build_dataset returns train_ds, val_ds)
    _, val_ds = build_dataset()

    # Load the trained model we saved in Step 4
    model_path = MODEL_OUTPUT_DIR / "final_stamp_classifier.keras"
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    print("Running Inference...")
    start_time = time.time()

    # Iterate over the validation batches and predict class probabilities
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        # Store true labels
        y_true.extend(labels.numpy())
        # Convert sigmoid probability output to binary class (0 or 1) using 0.5 threshold
        y_pred.extend((preds >= 0.5).astype(int).flatten())

    end_time = time.time()
    print(f"Total Inference Time: {end_time - start_time:.2f} seconds")

    # Print sklearn classification report
    # NOTE: Since the subfolders in BALANCED_ROI_DIR are loaded alphabetically,
    # Class 0 corresponds to "forged" and Class 1 to "genuine".
    # We corrected the target_names list to match this order! Otherwise, 
    # our precision/recall metrics would be flipped, which is a major facepalm lol.
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["forged", "genuine"]))

    # Print the confusion matrix: [[TN, FP], [FN, TP]]
    # Top-left is True Forged, bottom-right is True Genuine.
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate_model()

