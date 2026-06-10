# =====================================================================
# STAMP SEGMENTATION & COLOR THRESHOLDING
# This script contains the logic to create a binary mask of the stamp.
# We convert the BGR image to the HSV color space, which makes it much easier
# to isolate the specific color of stamp ink. Then we clean up noise 
# using morphological open and close operations.
# =====================================================================

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
    Applies precise HSV thresholding and morphology.
    Returns the processed binary mask.
    """
    # Calculate scale factor to resize image width to 900px while keeping aspect ratio.
    # Processing on a 900px wide image is way faster than on a giant raw scan!
    h0, w0 = image_bgr.shape[:2]
    scale = DISPLAY_WIDTH / w0
    new_h = int(h0 * scale)

    image_resized = cv2.resize(image_bgr, (DISPLAY_WIDTH, new_h))
    
    # Convert from BGR to HSV color space.
    # Why HSV? Because RGB is highly sensitive to lighting changes and shadows.
    # HSV (Hue, Saturation, Value) separates the actual color tint (Hue) 
    # from how colorful it is (Saturation) and how bright it is (Value).
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Split into H, S, V channels
    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]

    # Exact HSV ranges from our experimented values.
    # Blue/purple ink usually falls in Hue 90 to 170.
    hue_mask = cv2.inRange(h, 90, 170)
    sat_mask = cv2.inRange(s, 25, 255) # Ignore greyish/white backgrounds
    val_mask = cv2.inRange(v, 30, 255) # Ignore extremely dark/black shadows

    # Combine all masks together using bitwise AND operations
    mask = cv2.bitwise_and(hue_mask, sat_mask)
    mask = cv2.bitwise_and(mask, val_mask)

    # Exact Morphology kernels.
    # We use Elliptical structuring elements because stamps are circular/oval shaped.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # MORPH_OPEN (Erosion followed by Dilation)
    # This removes tiny white speckles (high-frequency noise or text ink bleed) from the mask.
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # MORPH_CLOSE (Dilation followed by Erosion)
    # This fills in tiny black gaps/cracks inside the stamp where the wet ink did not touch 
    # the paper. Ensures we get a solid, continuous shape for contour detection!
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

    return mask_cleaned

