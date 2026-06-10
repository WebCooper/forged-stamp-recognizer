import cv2
import numpy as np
from src.config import ROI_SIZE, DISPLAY_WIDTH


def contour_stamp_score(contour):
    """Scores candidates based on compactness, extent, and circularity (from Notebook 04)."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    if bbox_area <= 0:
        return None

    aspect_ratio = w / h
    compactness = min(w, h) / max(w, h)
    extent = area / bbox_area

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter**2)

    # Basic rejection rules
    if area < 200:
        return None
    if compactness < 0.35:
        return None

    # Weighted score (Fulfills the Geometric Localization requirement)
    score = 0.45 * compactness + 0.35 * circularity + 0.20 * extent

    return {"x": x, "y": y, "w": w, "h": h, "score": score}


def extract_roi(image_bgr, mask):
    """
    Extracts the highest scoring stamp, maps it back to the original
    high-resolution image, and crops it with a pad factor.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        result = contour_stamp_score(contour)
        if result is not None:
            candidates.append(result)

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)

    if not candidates:
        return None

    best = candidates[0]

    # Map back to original image coordinates
    original_h, original_w = image_bgr.shape[:2]
    resized_h = mask.shape[0]

    scale_x = original_w / DISPLAY_WIDTH
    scale_y = original_h / resized_h

    x, y, w, h = best["x"], best["y"], best["w"], best["h"]

    # Notebook 04 Padding Factor
    pad_factor = 1.0
    pad_x = int(w * pad_factor)
    pad_y = int(h * pad_factor)

    # Bounding box in resized coordinates
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, DISPLAY_WIDTH)
    y2 = min(y + h + pad_y, resized_h)

    # Convert to original coordinates
    ox1 = int(x1 * scale_x)
    oy1 = int(y1 * scale_y)
    ox2 = int(x2 * scale_x)
    oy2 = int(y2 * scale_y)

    roi_original = image_bgr[oy1:oy2, ox1:ox2]

    if roi_original.size == 0:
        return None

    # Standardize ROI to expected model size
    roi_resized = cv2.resize(
        roi_original, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA
    )

    return roi_resized
