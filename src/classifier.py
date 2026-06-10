"""
classifier.py
=============
Member 4 — Deep Learning & Evaluation Lead
EE7204 / EC7205 — Image Processing and Computer Vision
University of Ruhuna, Department of Electrical and Information Engineering

This module provides the complete deep learning pipeline for the
Image-Based Document Stamp Verification System. It encapsulates:
  - Model architecture construction (ResNet50 transfer learning)
  - Training data pipeline (tf.data with augmentation)
  - Two-phase training strategy (frozen base → fine-tuning)
  - Comprehensive evaluation (metrics, ROC, confusion matrix, Grad-CAM)
  - Inference utility for single-image prediction

Usage
-----
    from src.classifier import StampClassifier

    clf = StampClassifier()
    clf.build()
    clf.train(roi_dataset_root="outputs/roi_dataset_v3")
    clf.evaluate()
    clf.save("outputs/models/stamp_resnet50_final.keras")

    # Single-image inference
    result = clf.predict_single("path/to/stamp_roi.png")
    # Returns: {"class": "genuine", "confidence": 0.97, "inference_ms": 12.3}
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------------------------------------------------------------------
# Constants (can be overridden via StampClassifier constructor)
# ---------------------------------------------------------------------------
IMG_SIZE: int = 224          # ResNet50 expected input size
BATCH_SIZE: int = 16         # Suitable for ~300 image dataset
SEED: int = 42               # Reproducibility
CLASS_NAMES: dict = {0: "genuine", 1: "forged"}


class StampClassifier:
    """
    End-to-end stamp forgery classifier using ResNet50 transfer learning.

    Architecture
    ------------
    Input (224×224×3)
        → Data augmentation (rotation, zoom, translation, contrast)
        → ResNet50 backbone (ImageNet weights, frozen in Phase 1)
        → GlobalAveragePooling2D
        → Dropout(0.35)
        → Dense(128, relu)
        → Dropout(0.25)
        → Dense(1, sigmoid)  ← binary output: 0 = genuine, 1 = forged

    Training Strategy
    -----------------
    Phase 1 (frozen base): Train only the custom classification head.
        - lr = 1e-4, up to 25 epochs, early stopping (patience=6)
    Phase 2 (fine-tuning): Unfreeze top 30 layers of ResNet50 and retrain.
        - lr = 1e-5, up to 15 epochs, early stopping (patience=5)

    This two-phase approach prevents overfitting on the small custom dataset
    while allowing the top layers to adapt to microscopic ink texture features.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        batch_size: int = BATCH_SIZE,
        seed: int = SEED,
    ) -> None:
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

        self.model: tf.keras.Model | None = None
        self.base_model: tf.keras.Model | None = None
        self.history_phase1 = None
        self.history_phase2 = None

        # Split dataframes — populated by load_dataset()
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        # tf.data datasets — populated by _build_tf_datasets()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    # ------------------------------------------------------------------
    # 1. Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self, roi_dataset_root: str | Path) -> None:
        """
        Scan the ROI dataset directory and create stratified train/val/test splits.

        Expected directory structure
        ----------------------------
        roi_dataset_root/
            genuine/   (label = 0)
                *.png
            forged/    (label = 1)
                *.png

        Split ratios: 70% train · 15% validation · 15% test
        Stratification ensures equal class balance in every split.

        Parameters
        ----------
        roi_dataset_root : str or Path
            Root directory containing 'genuine' and 'forged' subdirectories.
        """
        roi_dataset_root = Path(roi_dataset_root)
        genuine_dir = roi_dataset_root / "genuine"
        forged_dir = roi_dataset_root / "forged"

        records = []
        for path in genuine_dir.rglob("*.png"):
            records.append({"image_path": str(path), "label": 0, "class_name": "genuine"})
        for path in forged_dir.rglob("*.png"):
            records.append({"image_path": str(path), "label": 1, "class_name": "forged"})

        df = pd.DataFrame(records)
        print(f"[Dataset] Total ROI images found: {len(df)}")
        print(df["class_name"].value_counts().to_string())

        # 70 / 15 / 15 stratified split
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label"], random_state=self.seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label"], random_state=self.seed
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        print(f"\n[Split] Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        self._build_tf_datasets()

    def _load_and_preprocess(self, image_path: tf.Tensor, label: tf.Tensor):
        """
        tf.data map function: read PNG → resize to 224×224 → ResNet50 preprocessing.

        ResNet50's preprocess_input converts RGB values to the zero-centred
        ImageNet mean-subtracted format the backbone expects.
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (self.img_size, self.img_size))
        image = preprocess_input(image)
        return image, label

    def _build_tf_datasets(self) -> None:
        """Build tf.data pipelines with shuffling and prefetching for performance."""

        def make_ds(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices(
                (df["image_path"].values, df["label"].values)
            )
            if shuffle:
                ds = ds.shuffle(buffer_size=len(df), seed=self.seed)
            ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return ds

        self.train_ds = make_ds(self.train_df, shuffle=True)
        self.val_ds = make_ds(self.val_df)
        self.test_ds = make_ds(self.test_df)

    # ------------------------------------------------------------------
    # 2. Model architecture
    # ------------------------------------------------------------------

    def build(self) -> tf.keras.Model:
        """
        Construct the transfer learning model.

        Architecture justification
        --------------------------
        ResNet50 was selected over VGG16 because its residual skip connections
        allow gradients to flow more cleanly during fine-tuning on a small
        dataset. VGG16's fully-connected layers also carry significantly more
        parameters, increasing overfitting risk.

        The custom head uses two Dropout layers to regularise the small dataset,
        and a single sigmoid output for binary (genuine vs forged) classification.

        Returns
        -------
        tf.keras.Model
            The compiled model (Phase 1 configuration).
        """
        # Augmentation applied only during training (training=True in model.fit)
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomRotation(0.08),         # ±29° — simulates orientation variance
                layers.RandomZoom(0.08),              # ±8% zoom — simulates scan distance
                layers.RandomTranslation(0.05, 0.05),# ±5% shift — simulates stamp placement
                layers.RandomContrast(0.15),          # ±15% contrast — simulates illumination
                layers.RandomFlip("horizontal"),      # Stamps can appear mirrored
            ],
            name="data_augmentation",
        )

        # ResNet50 backbone — frozen for Phase 1
        self.base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
        )
        self.base_model.trainable = False

        # Build functional API model
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3), name="input_image")
        x = data_augmentation(inputs)
        x = self.base_model(x, training=False)          # training=False keeps BN in inference mode
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dropout(0.35, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.25, name="dropout_2")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        self.model = models.Model(inputs, outputs, name="StampClassifier_ResNet50")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        print(f"[Model] Built. Trainable parameters: {self.model.count_params():,}")
        return self.model

    # ------------------------------------------------------------------
    # 3. Training
    # ------------------------------------------------------------------

    def _compute_class_weights(self) -> dict:
        """
        Compute class weights to handle potential class imbalance.

        If genuine and forged classes have different counts in the training
        set, the loss function penalises the majority class less — preventing
        the model from biasing predictions toward the larger class.
        """
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=self.train_df["label"].values,
        )
        cw = {0: weights[0], 1: weights[1]}
        print(f"[Class weights] Genuine: {cw[0]:.3f} | Forged: {cw[1]:.3f}")
        return cw

    def train_phase1(
        self,
        epochs: int = 25,
        model_dir: str | Path = "outputs/models",
    ) -> None:
        """
        Phase 1 — Train only the custom classification head (frozen backbone).

        The ResNet50 base remains frozen. Only the Dense layers above it are
        updated. This fast phase lets the head learn to interpret ResNet50's
        feature maps for stamp texture.

        Parameters
        ----------
        epochs : int
            Maximum training epochs (early stopping may terminate sooner).
        model_dir : str or Path
            Directory to save the best model checkpoint.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        class_weight = self._compute_class_weights()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_dir / "stamp_resnet50_phase1_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        print("\n[Phase 1] Training classification head (backbone frozen)...")
        self.history_phase1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    def train_phase2(
        self,
        epochs: int = 20,
        fine_tune_from_layer: int = -30,
        model_dir: str | Path = "outputs/models",
    ) -> None:
        """
        Phase 2 — Fine-tune the top layers of the ResNet50 backbone.

        Unfreezes the top `abs(fine_tune_from_layer)` layers of ResNet50 and
        retrains with a very low learning rate (1e-5). This allows the upper
        convolutional blocks to adjust their filters towards microscopic ink
        texture differences (wet-ink diffusion vs halftone dot patterns).

        A ReduceLROnPlateau callback further lowers the rate if the validation
        loss plateaus, preventing destructive updates to the pre-trained weights.

        Parameters
        ----------
        epochs : int
            Maximum fine-tuning epochs.
        fine_tune_from_layer : int
            Negative index into base_model.layers. Default -30 unfreezes the
            top 30 layers (the last two residual blocks of ResNet50).
        model_dir : str or Path
            Directory to save the best fine-tuned model checkpoint.
        """
        model_dir = Path(model_dir)
        class_weight = self._compute_class_weights()

        # Unfreeze top layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:fine_tune_from_layer]:
            layer.trainable = False

        trainable_count = sum(1 for l in self.base_model.layers if l.trainable)
        print(f"\n[Phase 2] Unfroze top {trainable_count} backbone layers for fine-tuning.")

        # Recompile with low learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_dir / "stamp_resnet50_phase2_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1,
            ),
        ]

        print("[Phase 2] Fine-tuning (very low lr=1e-5)...")
        self.history_phase2 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    def train(
        self,
        roi_dataset_root: str | Path,
        model_dir: str | Path = "outputs/models",
        phase1_epochs: int = 25,
        phase2_epochs: int = 20,
    ) -> None:
        """
        Convenience wrapper: load dataset → build model → Phase 1 → Phase 2.

        Parameters
        ----------
        roi_dataset_root : str or Path
            Root of the extracted ROI dataset (genuine/ + forged/).
        model_dir : str or Path
            Where to save model checkpoints.
        phase1_epochs : int
            Max epochs for frozen-head training.
        phase2_epochs : int
            Max epochs for fine-tuning.
        """
        self.load_dataset(roi_dataset_root)
        self.build()
        self.train_phase1(epochs=phase1_epochs, model_dir=model_dir)
        self.train_phase2(epochs=phase2_epochs, model_dir=model_dir)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, figures_dir: str | Path = "outputs/figures") -> dict:
        """
        Run comprehensive evaluation on the held-out test set.

        Produces
        --------
        - Console: accuracy, precision, recall, F1-score, AUC
        - Figure 1: Combined training curves (Phase 1 + Phase 2)
        - Figure 2: Confusion matrix
        - Figure 3: ROC curve with AUC annotation
        - Figure 4: Precision-Recall curve with AP annotation
        - Figure 5: Misclassified samples gallery
        - Figure 6: Inference speed distribution

        Returns
        -------
        dict
            Scalar metric summary {accuracy, precision, recall, f1, auc, ...}
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # --- Predictions ---
        y_true = self.test_df["label"].values
        y_prob = self.model.predict(self.test_ds, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        # --- Scalar metrics ---
        report = classification_report(
            y_true, y_pred, target_names=["genuine", "forged"], output_dict=True
        )
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        accuracy   = report["accuracy"]
        precision  = report["forged"]["precision"]
        recall     = report["forged"]["recall"]
        f1         = report["forged"]["f1-score"]

        print("\n" + "=" * 52)
        print("  EVALUATION RESULTS — TEST SET")
        print("=" * 52)
        print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  Avg Prec  : {ap:.4f}")
        print("=" * 52)
        print("\nPer-class report:")
        print(classification_report(y_true, y_pred, target_names=["genuine", "forged"]))

        # --- Inference speed ---
        inf_times = self._measure_inference_speed()
        mean_ms = np.mean(inf_times)
        print(f"\n[Inference] Mean: {mean_ms:.1f} ms/image | "
              f"Std: {np.std(inf_times):.1f} ms | "
              f"Max: {np.max(inf_times):.1f} ms")

        # --- Figures ---
        self._plot_training_curves(figures_dir)
        self._plot_confusion_matrix(y_true, y_pred, figures_dir)
        self._plot_roc_curve(y_true, y_prob, auc, figures_dir)
        self._plot_pr_curve(y_true, y_prob, ap, figures_dir)
        self._plot_misclassified(y_true, y_pred, y_prob, figures_dir)
        self._plot_inference_speed(inf_times, figures_dir)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "average_precision": ap,
            "inference_mean_ms": mean_ms,
        }

    def _measure_inference_speed(self, n_runs: int = 30) -> list[float]:
        """
        Measure per-image inference time on the test set.

        Each image is processed individually (batch_size=1) to get a realistic
        single-document inference time. The first call is a warm-up (excluded).
        """
        image_paths = self.test_df["image_path"].values
        times_ms = []

        # Warm-up run
        sample = self._load_single_image(image_paths[0])
        self.model.predict(sample, verbose=0)

        n = min(n_runs, len(image_paths))
        for path in image_paths[:n]:
            img = self._load_single_image(path)
            t0 = time.perf_counter()
            self.model.predict(img, verbose=0)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        return times_ms

    def _load_single_image(self, path: str) -> np.ndarray:
        """Load, resize, and preprocess a single image for inference."""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = preprocess_input(img.astype(np.float32))
        return np.expand_dims(img, axis=0)

    # ------------------------------------------------------------------
    # 5. Plotting helpers
    # ------------------------------------------------------------------

    def _plot_training_curves(self, out_dir: Path) -> None:
        """Plot combined Phase 1 + Phase 2 accuracy and loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def get_hist(h):
            return h.history if h is not None else {}

        def combine(key):
            p1 = get_hist(self.history_phase1).get(key, [])
            p2 = get_hist(self.history_phase2).get(key, [])
            return p1 + p2

        acc = combine("accuracy")
        val_acc = combine("val_accuracy")
        loss = combine("loss")
        val_loss = combine("val_loss")

        phase_boundary = len(get_hist(self.history_phase1).get("accuracy", []))

        ax = axes[0]
        ax.plot(acc, label="Train", color="#2563eb")
        ax.plot(val_acc, label="Validation", color="#16a34a")
        if phase_boundary:
            ax.axvline(phase_boundary, color="gray", linestyle="--", linewidth=1,
                       label=f"Fine-tuning starts (ep {phase_boundary})")
        ax.set_title("Accuracy — Phase 1 + Phase 2", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(loss, label="Train", color="#2563eb")
        ax.plot(val_loss, label="Validation", color="#16a34a")
        if phase_boundary:
            ax.axvline(phase_boundary, color="gray", linestyle="--", linewidth=1,
                       label=f"Fine-tuning starts (ep {phase_boundary})")
        ax.set_title("Loss — Phase 1 + Phase 2", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("Training History — StampClassifier (ResNet50)", fontsize=14, y=1.02)
        plt.tight_layout()
        path = out_dir / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    def _plot_confusion_matrix(self, y_true, y_pred, out_dir: Path) -> None:
        """Plot and save the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Genuine", "Forged"]
        )
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix — Test Set", fontsize=13)

        # Annotate TN/FP/FN/TP labels
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i + 0.3, labels[i][j],
                    ha="center", va="center",
                    fontsize=10, color="gray", alpha=0.7,
                )
        plt.tight_layout()
        path = out_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    def _plot_roc_curve(self, y_true, y_prob, auc: float, out_dir: Path) -> None:
        """Plot the ROC curve with AUC score."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.set_title("ROC Curve — Stamp Forgery Classifier", fontsize=13)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / "roc_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    def _plot_pr_curve(self, y_true, y_prob, ap: float, out_dir: Path) -> None:
        """Plot the Precision-Recall curve (more informative for imbalanced datasets)."""
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall_vals, precision_vals, color="#16a34a", lw=2,
                label=f"PR Curve (AP = {ap:.4f})")
        baseline = y_true.sum() / len(y_true)
        ax.axhline(baseline, color="gray", linestyle="--", lw=1,
                   label=f"Random classifier (AP ≈ {baseline:.2f})")
        ax.fill_between(recall_vals, precision_vals, alpha=0.1, color="#16a34a")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontsize=13)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / "pr_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    def _plot_misclassified(self, y_true, y_pred, y_prob, out_dir: Path) -> None:
        """
        Display a gallery of misclassified test samples.

        Showing misclassified samples gives forensic insight into failure modes:
        - FP (genuine predicted as forged): likely faded or noisy genuine stamps
        - FN (forged predicted as genuine): high-quality forgeries that fool the model
        """
        errors = np.where(y_true != y_pred)[0]
        if len(errors) == 0:
            print("[Info] No misclassified samples — perfect test set accuracy!")
            return

        n = min(12, len(errors))
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = np.array(axes).flatten()

        for idx, err_idx in enumerate(errors[:n]):
            path = self.test_df["image_path"].values[err_idx]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            true_label = CLASS_NAMES[y_true[err_idx]]
            pred_label = CLASS_NAMES[y_pred[err_idx]]
            conf = y_prob[err_idx] if y_pred[err_idx] == 1 else 1 - y_prob[err_idx]

            axes[idx].imshow(img)
            axes[idx].set_title(
                f"True: {true_label}\nPred: {pred_label} ({conf:.2f})",
                fontsize=9,
                color="red",
            )
            axes[idx].axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        plt.suptitle(
            f"Misclassified Samples ({len(errors)} total) — Test Set",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        path = out_dir / "misclassified_gallery.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    def _plot_inference_speed(self, times_ms: list[float], out_dir: Path) -> None:
        """Plot inference speed distribution with statistics annotated."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(times_ms, bins=15, color="#7c3aed", alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(times_ms), color="red", linestyle="--", lw=1.5,
                   label=f"Mean: {np.mean(times_ms):.1f} ms")
        ax.axvline(np.median(times_ms), color="orange", linestyle="--", lw=1.5,
                   label=f"Median: {np.median(times_ms):.1f} ms")
        ax.set_xlabel("Inference time (ms per image)")
        ax.set_ylabel("Count")
        ax.set_title("Inference Speed Distribution — Single Image, CPU", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / "inference_speed.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path}")

    # ------------------------------------------------------------------
    # 6. Grad-CAM — explainability
    # ------------------------------------------------------------------

    def grad_cam(self, image_path: str, layer_name: str = "conv5_block3_out") -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which
        regions of the stamp the model attended to when making its decision.
        This is critical for forensic validation — we can confirm the model
        is looking at ink texture regions, not document background artefacts.

        Parameters
        ----------
        image_path : str
            Path to a 224×224 ROI image.
        layer_name : str
            Name of the target convolutional layer. For ResNet50, the last
            convolutional layer is 'conv5_block3_out'.

        Returns
        -------
        np.ndarray
            BGR image with Grad-CAM heatmap overlaid (for cv2.imshow / cv2.imwrite).
        """
        # Build Grad-CAM model that outputs both the target layer and the final prediction
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.base_model.name).get_layer(layer_name).output,
                self.model.output,
            ],
        )

        # Load and preprocess image
        img_array = self._load_single_image(image_path)
        img_tensor = tf.cast(img_array, tf.float32)

        # Record gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            # For binary sigmoid: gradient of prediction w.r.t. conv feature maps
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap and overlay on original image
        original = cv2.imread(image_path)
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlaid = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

        return overlaid

    def plot_grad_cam_gallery(
        self,
        n_genuine: int = 3,
        n_forged: int = 3,
        figures_dir: str | Path = "outputs/figures",
    ) -> None:
        """
        Plot Grad-CAM heatmaps for sample genuine and forged stamps side-by-side.

        Useful for demonstrating model interpretability in the project report —
        the heatmaps should concentrate on stamp ink texture regions, not
        surrounding document text or blank areas.
        """
        figures_dir = Path(figures_dir)
        genuine_paths = self.test_df[self.test_df["label"] == 0]["image_path"].values[:n_genuine]
        forged_paths  = self.test_df[self.test_df["label"] == 1]["image_path"].values[:n_forged]

        total = n_genuine + n_forged
        fig, axes = plt.subplots(2, total, figsize=(3.5 * total, 7))

        for col, path in enumerate(list(genuine_paths) + list(forged_paths)):
            original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cam = cv2.cvtColor(self.grad_cam(path), cv2.COLOR_BGR2RGB)
            label = "Genuine" if col < n_genuine else "Forged"

            axes[0, col].imshow(original)
            axes[0, col].set_title(f"{label}\nOriginal ROI", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(cam)
            axes[1, col].set_title("Grad-CAM\nheatmap", fontsize=9)
            axes[1, col].axis("off")

        plt.suptitle("Grad-CAM Explainability — Model Attention on Stamp ROIs", fontsize=13)
        plt.tight_layout()
        path_out = figures_dir / "gradcam_gallery.png"
        plt.savefig(path_out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Saved] {path_out}")

    # ------------------------------------------------------------------
    # 7. Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = "outputs/models/stamp_resnet50_final.keras") -> None:
        """Save the trained model to disk in Keras format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"[Saved] Model → {path.resolve()}")

    @classmethod
    def load(cls, path: str | Path) -> "StampClassifier":
        """
        Load a previously saved StampClassifier model.

        Parameters
        ----------
        path : str or Path
            Path to a .keras model file.

        Returns
        -------
        StampClassifier
            Instance with the loaded model ready for inference or continued training.
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(str(path))
        print(f"[Loaded] Model ← {Path(path).resolve()}")
        return instance

    # ------------------------------------------------------------------
    # 8. Single-image inference
    # ------------------------------------------------------------------

    def predict_single(self, image_path: str | Path) -> dict:
        """
        Run inference on a single stamp ROI image.

        This is the function that would be called in a production document
        verification system after the ROI extraction pipeline (Members 2 & 3)
        has cropped the stamp region.

        Parameters
        ----------
        image_path : str or Path
            Path to a stamp ROI image (any resolution; will be resized).

        Returns
        -------
        dict with keys:
            class       : "genuine" or "forged"
            confidence  : float in [0, 1] — probability of the predicted class
            raw_prob    : float — raw sigmoid output (probability of "forged")
            inference_ms: float — inference time in milliseconds
        """
        img = self._load_single_image(str(image_path))
        t0 = time.perf_counter()
        prob_forged = float(self.model.predict(img, verbose=0)[0, 0])
        t1 = time.perf_counter()

        predicted_class = 1 if prob_forged >= 0.5 else 0
        confidence = prob_forged if predicted_class == 1 else 1.0 - prob_forged

        return {
            "class": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "raw_prob": round(prob_forged, 4),
            "inference_ms": round((t1 - t0) * 1000, 2),
        }
