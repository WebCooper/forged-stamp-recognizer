import cv2
from src.config import ROI_SIZE


def extract_roi(image_bgr, mask):
    """Finds the geometric center of the stamp and crops an ROI."""
    # Canny Edge Detection to define boundaries
    edges = cv2.Canny(mask, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Assume the largest contour by area is the stamp
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate center
    center_x, center_y = x + w // 2, y + h // 2

    # Calculate crop coordinates ensuring a square ROI
    half_size = ROI_SIZE // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(image_bgr.shape[1], center_x + half_size)
    y2 = min(image_bgr.shape[0], center_y + half_size)

    roi = image_bgr[y1:y2, x1:x2]

    # Pad if ROI is smaller than expected due to image borders
    if roi.shape[:2] != (ROI_SIZE, ROI_SIZE):
        roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))

    return roi
