"""
inference.py
============
Member 4 — Deep Learning & Evaluation Lead
EE7204 / EC7205 — Image Processing and Computer Vision
University of Ruhuna

Standalone inference script. Given a raw document image (not a pre-cropped ROI),
this script runs the FULL pipeline:
  1. Load and resize the document image
  2. HSV segmentation + morphological cleaning  (re-implements Member 2's logic)
  3. Contour-based ROI extraction               (re-implements Member 3's logic)
  4. ResNet50 classifier                        (Member 4)
  5. Print verdict + confidence

Usage
-----
    python src/inference.py --image path/to/document.png \\
                            --model outputs/models/stamp_resnet50_final.keras

Or import as a module:
    from src.inference import run_full_pipeline
    result = run_full_pipeline("document.png", "model.keras")
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ── Constants ────────────────────────────────────────────────────────────────
DISPLAY_WIDTH = 900     # Resize document to this width for processing
IMG_SIZE      = 224     # ResNet50 input size
CLASS_NAMES   = {0: "GENUINE", 1: "FORGED"}
COLORS        = {0: (0, 200, 80), 1: (0, 60, 220)}   # BGR: green / red


# ── Step 1: Segmentation (Member 2's pipeline, re-implemented for inference) ─

def segment_stamp_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to HSV, threshold for violet/blue ink, apply morphological cleaning.

    This reproduces the logic from notebook 02 / notebook 03.
    HSV range [90–170] captures violet and blue ink hues.
    Saturation threshold [25, 255] removes achromatic document text.

    Returns a binary mask (0/255) of the stamp region.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]

    hue_mask = cv2.inRange(h, 90, 170)
    sat_mask = cv2.inRange(s, 25, 255)
    val_mask = cv2.inRange(v, 30, 255)   # avoid very dark noise
    mask = cv2.bitwise_and(hue_mask, cv2.bitwise_and(sat_mask, val_mask))

    # Morphological closing: fill gaps caused by uneven ink absorption
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Morphological opening: remove thin text/noise structures
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


# ── Step 2: ROI extraction (Member 3's pipeline, re-implemented) ─────────────

def _contour_stamp_score(contour) -> dict | None:
    """
    Score a contour candidate on how likely it is to be a stamp.
    Mirrors the scoring function from notebooks 03–04.
    """
    area = cv2.contourArea(contour)
    if area < 500:         # too small
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    if bbox_area <= 0:
        return None

    aspect_ratio = w / h
    compactness  = min(w, h) / max(w, h)
    extent       = area / bbox_area

    # Reward compact (near-square), reasonably large, well-filled contours
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return None

    score = (compactness * 0.4 + extent * 0.3 + min(area / 5000, 1.0) * 0.3)
    return {"contour": contour, "x": x, "y": y, "w": w, "h": h, "score": score}


def extract_stamp_roi(
    original_bgr: np.ndarray,
    display_width: int = DISPLAY_WIDTH,
    pad_factor: float = 0.8,
) -> tuple[np.ndarray | None, dict | None]:
    """
    Run the full segmentation + localization pipeline on a document image.

    Parameters
    ----------
    original_bgr : np.ndarray
        Full-resolution document scan (BGR).
    display_width : int
        Width to resize the document to for processing.
    pad_factor : float
        Extra padding around the detected stamp bounding box.

    Returns
    -------
    roi_bgr : np.ndarray or None
        Cropped stamp region (224×224) or None if no stamp was found.
    info : dict or None
        Detection metadata (bbox, score, confidence) or None.
    """
    h0, w0 = original_bgr.shape[:2]
    scale   = display_width / w0
    new_h   = int(h0 * scale)

    # Resize for processing
    image_resized = cv2.resize(original_bgr, (display_width, new_h))

    # Segment
    mask = segment_stamp_mask(image_resized)

    # Find and score contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []
    for cnt in contours:
        result = _contour_stamp_score(cnt)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return None, None

    # Take the highest-scoring contour
    best = sorted(candidates, key=lambda c: c["score"], reverse=True)[0]

    # Scale bbox back to original resolution
    inv = 1.0 / scale
    x  = int(best["x"] * inv)
    y  = int(best["y"] * inv)
    w  = int(best["w"] * inv)
    h  = int(best["h"] * inv)

    # Padded crop
    px = int(w * pad_factor)
    py = int(h * pad_factor)
    x1 = max(x - px, 0)
    y1 = max(y - py, 0)
    x2 = min(x + w + px, w0)
    y2 = min(y + h + py, h0)

    roi_bgr = original_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None

    roi_bgr = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    info = {
        "bbox_original": (x1, y1, x2, y2),
        "score": best["score"],
    }
    return roi_bgr, info


# ── Step 3: Classification ────────────────────────────────────────────────────

def classify_roi(model: tf.keras.Model, roi_bgr: np.ndarray) -> dict:
    """
    Run the trained ResNet50 classifier on a 224×224 BGR stamp ROI.

    Returns
    -------
    dict with keys: class, confidence, raw_prob, inference_ms
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img     = preprocess_input(roi_rgb.astype(np.float32))
    img     = np.expand_dims(img, axis=0)

    t0            = time.perf_counter()
    prob_forged   = float(model.predict(img, verbose=0)[0, 0])
    inference_ms  = (time.perf_counter() - t0) * 1000

    predicted_cls = 1 if prob_forged >= 0.5 else 0
    confidence    = prob_forged if predicted_cls == 1 else 1.0 - prob_forged

    return {
        "class"        : CLASS_NAMES[predicted_cls],
        "class_id"     : predicted_cls,
        "confidence"   : round(confidence, 4),
        "raw_prob"     : round(prob_forged, 4),
        "inference_ms" : round(inference_ms, 2),
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_path: str,
    model_path: str,
    save_annotated: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the complete end-to-end stamp verification pipeline on a document image.

    Parameters
    ----------
    image_path : str
        Path to the input document image (PNG, JPG, etc.).
    model_path : str
        Path to the trained .keras model file.
    save_annotated : str or None
        If provided, saves an annotated version of the document with the
        detected stamp highlighted and the verdict overlaid.
    verbose : bool
        If True, print results to console.

    Returns
    -------
    dict
        {
          "status"      : "ok" or "error",
          "class"       : "GENUINE" or "FORGED",
          "confidence"  : float,
          "raw_prob"    : float,
          "inference_ms": float,
          "message"     : str
        }
    """
    t_total_start = time.perf_counter()

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {"status": "error", "message": f"Could not load image: {image_path}"}

    # Load model
    model = tf.keras.models.load_model(str(model_path))

    # Extract ROI
    roi_bgr, roi_info = extract_stamp_roi(image_bgr)
    if roi_bgr is None:
        return {
            "status" : "error",
            "message": "No stamp region detected in the document. "
                       "Check that the document contains a visible colored stamp.",
        }

    # Classify
    result = classify_roi(model, roi_bgr)
    total_ms = (time.perf_counter() - t_total_start) * 1000

    result["status"]     = "ok"
    result["total_ms"]   = round(total_ms, 2)
    result["image_path"] = str(image_path)

    if verbose:
        print()
        print("=" * 52)
        print("  STAMP VERIFICATION RESULT")
        print("=" * 52)
        print(f"  Image        : {Path(image_path).name}")
        print(f"  Verdict      : {result['class']}")
        print(f"  Confidence   : {result['confidence']*100:.1f}%")
        print(f"  P(forged)    : {result['raw_prob']:.4f}")
        print(f"  Inference    : {result['inference_ms']:.1f} ms")
        print(f"  Total time   : {total_ms:.1f} ms (incl. I/O + segmentation)")
        print("=" * 52)

    # Save annotated image
    if save_annotated:
        annotated = image_bgr.copy()
        x1, y1, x2, y2 = roi_info["bbox_original"]
        color = COLORS[result["class_id"]]
        label = f"{result['class']} ({result['confidence']*100:.0f}%)"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(save_annotated, annotated)
        if verbose:
            print(f"  Annotated    : {save_annotated}")

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stamp Forgery Detector — EE7204/EC7205, University of Ruhuna"
    )
    parser.add_argument("--image", required=True, help="Path to document image")
    parser.add_argument("--model", required=True, help="Path to .keras model file")
    parser.add_argument("--save",  default=None,  help="Save annotated image to this path")
    args = parser.parse_args()

    run_full_pipeline(
        image_path=args.image,
        model_path=args.model,
        save_annotated=args.save,
        verbose=True,
    )
