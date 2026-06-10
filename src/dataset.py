# =====================================================================
# DATA BALANCING & AUGMENTATION SCRIPT
# Deep learning models need lots of data. Since our manual cleaning deletes
# some bad crops, we end up with fewer images. Also, the genuine and forged classes
# might have unequal counts. This script balances them and augments the images to 
# make our dataset larger and robust against orientation, noise, and lighting!
# =====================================================================

import cv2
import numpy as np
import random
from src.config import RAW_ROI_DIR, BALANCED_ROI_DIR, TARGET_CLASS_SIZE


def augment_roi(roi, num_variations):
    """
    Applies random augmentations to an ROI image to generate multiple variations.
    The first variation is always the original image.
    """
    augmentations = [roi]
    if num_variations <= 1:
        return augmentations

    for _ in range(num_variations - 1):
        # Pick one augmentation type randomly
        aug_type = random.choice(["rotate", "noise", "contrast"])
        
        if aug_type == "rotate":
            # Randomly select a 90, 180, or 270 degree rotation
            angle = random.choice(
                [
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ]
            )
            augmentations.append(cv2.rotate(roi, angle))
            
        elif aug_type == "noise":
            # Generate Gaussian noise (mean=0, std=15) matching image shape
            noise = np.random.normal(0, 15, roi.shape).astype(np.uint8)
            # Use cv2.add instead of '+' because cv2.add clips values at 255.
            # Plain numpy '+' would wrap around (e.g. 250 + 10 = 4) which ruins the image!
            augmentations.append(cv2.add(roi, noise))
            
        elif aug_type == "contrast":
            # Randomly adjust contrast (alpha) and brightness (beta)
            alpha = random.uniform(0.8, 1.3)
            beta = random.randint(-30, 30)
            # cv2.convertScaleAbs applies: output_pixel = input_pixel * alpha + beta
            # It also automatically handles clipping to [0, 255]
            augmentations.append(cv2.convertScaleAbs(roi, alpha=alpha, beta=beta))
            
    return augmentations


def generate_balanced_dataset():
    """
    Reads the clean crops, calculates how many variations are needed per image,
    and saves exactly TARGET_CLASS_SIZE (1200) images per folder.
    """
    print(f"Reading Cleaned ROIs from: {RAW_ROI_DIR}")

    for folder in ["genuine", "forged"]:
        img_paths = list((RAW_ROI_DIR / folder).glob("*.png"))
        num_surviving = len(img_paths)

        if num_surviving == 0:
            continue

        # Calculate exact math to hit the TARGET_CLASS_SIZE perfectly.
        # base_multiplier is how many copies of EACH image we make.
        # remainder is the leftover count we distribute to reach the target.
        # E.g., if target is 1200 and we have 500 images:
        # 1200 // 500 = 2, remainder = 200.
        # So 200 images get 3 copies, and 300 images get 2 copies. Total = 200*3 + 300*2 = 1200!
        base_multiplier = TARGET_CLASS_SIZE // num_surviving
        remainder = TARGET_CLASS_SIZE % num_surviving

        print(f"Processing '{folder}': Found {num_surviving} clean images.")
        print(f" -> Target size is {TARGET_CLASS_SIZE}. Distributing exactly...")

        for idx, path in enumerate(img_paths):
            roi = cv2.imread(str(path))
            if roi is not None:
                # Add 1 extra variation to the first 'remainder' number of images
                # to satisfy the exact class size balance.
                current_variations = base_multiplier + (1 if idx < remainder else 0)

                # Generate the variations
                augmented_rois = augment_roi(roi, num_variations=current_variations)

                # Save them to disk
                for aug_idx, aug_img in enumerate(augmented_rois):
                    out_path = BALANCED_ROI_DIR / folder / f"{path.stem}_{aug_idx}.png"
                    cv2.imwrite(str(out_path), aug_img)


if __name__ == "__main__":
    generate_balanced_dataset()

