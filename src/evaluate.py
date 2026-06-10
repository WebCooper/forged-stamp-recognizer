import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from src.config import MODEL_OUTPUT_DIR
from src.train import build_dataset


def evaluate_model():
    # Load validation data
    _, val_ds = build_dataset()

    # Load model
    model_path = MODEL_OUTPUT_DIR / "final_stamp_classifier.keras"
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    print("Running Inference...")
    start_time = time.time()

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend((preds >= 0.5).astype(int).flatten())

    end_time = time.time()
    print(f"Total Inference Time: {end_time - start_time:.2f} seconds")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["genuine", "forged"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate_model()
