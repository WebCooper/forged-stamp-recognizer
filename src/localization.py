# =====================================================================
# GEOMETRIC STAMP LOCALIZATION & CROPPING
# This script contains the logic to locate stamp candidates in the mask.
# Since we consider circular stamps, we score every contour based on
# geometric properties (circularity, compactness, extent). The highest-scoring
# contour is cropped out of the original high-resolution image.
# =====================================================================

import cv2
import numpy as np
from src.config import ROI_SIZE, DISPLAY_WIDTH


def contour_stamp_score(contour):
    """
    Scores candidates based on compactness, extent, and circularity (from Notebook 04).
    Returns a dictionary of bounding box coordinates and the score, or None if rejected.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    if bbox_area <= 0:
        return None

    aspect_ratio = w / h
    # Compactness measures how square/even the shape is. Close to 1.0 is a square/circle,
    # while close to 0 is a thin, long rectangle.
    compactness = min(w, h) / max(w, h)
    # Extent measures how much of the bounding box is occupied by the contour.
    # For a perfect circle, extent is area_circle / area_square = (pi * r^2) / (4 * r^2) ≈ 0.785
    extent = area / bbox_area

    # Calculate circularity (Metric = 4 * pi * Area / Perimeter^2)
    # A perfect circle has circularity = 1.0. Star or line will be close to 0.0.
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter**2)

    # Basic rejection rules: reject tiny noise spots or extremely elongated lines
    if area < 200:
        return None
    if compactness < 0.35:
        return None

    # Weighted score (Fulfills the Geometric Localization requirement)
    # Compactness and circularity are weighted higher because stamps are circular.
    score = 0.45 * compactness + 0.35 * circularity + 0.20 * extent

    return {"x": x, "y": y, "w": w, "h": h, "score": score}


def extract_roi(image_bgr, mask):
    """
    Extracts the highest scoring stamp, maps it back to the original
    high-resolution image, and crops it with a pad factor.
    """
    # Find outer contours of the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        result = contour_stamp_score(contour)
        if result is not None:
            candidates.append(result)

    # Sort candidates by score in descending order (highest score first)
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)

    if not candidates:
        return None

    # Grab the best candidate (the actual stamp)
    best = candidates[0]

    # Map back to original image coordinates.
    # Since the mask was generated on a resized image (900px wide), we calculate
    # scaling ratios to crop the stamp out of the ORIGINAL high-resolution scan.
    # This preserves the microscopic ink texture which our ResNet model needs!
    original_h, original_w = image_bgr.shape[:2]
    resized_h = mask.shape[0]

    scale_x = original_w / DISPLAY_WIDTH
    scale_y = original_h / resized_h

    x, y, w, h = best["x"], best["y"], best["w"], best["h"]

    #  Padding Factor: 
    # We add 1.0 * w/h padding so we don't crop the edges of the stamp too tightly.
    pad_factor = 1.0
    pad_x = int(w * pad_factor)
    pad_y = int(h * pad_factor)

    # Calculate bounding box in resized coordinates with padding, making sure
    # to clip inside image boundaries so we don't cause crop index errors.
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, DISPLAY_WIDTH)
    y2 = min(y + h + pad_y, resized_h)

    # Scale the coordinates to the original high-resolution image size
    ox1 = int(x1 * scale_x)
    oy1 = int(y1 * scale_y)
    ox2 = int(x2 * scale_x)
    oy2 = int(y2 * scale_y)

    # Crop the stamp from the original high-resolution image
    roi_original = image_bgr[oy1:oy2, ox1:ox2]

    if roi_original.size == 0:
        return None

    # Standardize the cropped patch to the size expected by ResNet50 (224x224).
    # We use cv2.INTER_AREA because it's the best interpolation method for shrinking 
    # images without adding aliasing artifacts.
    roi_resized = cv2.resize(
        roi_original, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA
    )

    return roi_resized

