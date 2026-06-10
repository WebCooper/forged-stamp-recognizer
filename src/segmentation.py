import cv2
import numpy as np
from src.config import DISPLAY_WIDTH


def preprocess_image(image_path):
    """Loads the original image. Resizing is handled dynamically in the next steps."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    return image_bgr


def segment_stamp(image_bgr):
    """
    Applies precise HSV thresholding and morphology from Notebook 04.
    Returns the processed mask.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = DISPLAY_WIDTH / w0
    new_h = int(h0 * scale)

    image_resized = cv2.resize(image_bgr, (DISPLAY_WIDTH, new_h))
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]

    # Exact HSV ranges from Notebook 04
    hue_mask = cv2.inRange(h, 90, 170)
    sat_mask = cv2.inRange(s, 25, 255)
    val_mask = cv2.inRange(v, 30, 255)

    mask = cv2.bitwise_and(hue_mask, sat_mask)
    mask = cv2.bitwise_and(mask, val_mask)

    # Exact Morphology kernels from Notebook 04
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

    return mask_cleaned
