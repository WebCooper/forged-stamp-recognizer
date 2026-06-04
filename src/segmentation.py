import cv2
import numpy as np
from src.config import DISPLAY_WIDTH


def preprocess_image(image_path):
    """Loads and resizes the image to standardize processing."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")

    h, w = image_bgr.shape[:2]
    scale = DISPLAY_WIDTH / w
    new_h = int(h * scale)
    return cv2.resize(image_bgr, (DISPLAY_WIDTH, new_h))


def segment_stamp(image_bgr):
    """Applies HSV thresholding and morphological operations to isolate the stamp."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define color range for typical stamp ink (e.g., blue/purple/red)
    # Adjust these bounds based on your specific dataset
    lower_bound = np.array([90, 50, 50])
    upper_bound = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Mathematical Morphology (Closing then Opening with a circular structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return clean_mask
