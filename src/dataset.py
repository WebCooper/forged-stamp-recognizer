import cv2
import numpy as np
import random
from src.config import RAW_ROI_DIR, BALANCED_ROI_DIR, TARGET_CLASS_SIZE


def augment_roi(roi, num_variations):
    augmentations = [roi]
    if num_variations <= 1:
        return augmentations

    for _ in range(num_variations - 1):
        aug_type = random.choice(["rotate", "noise", "contrast"])
        if aug_type == "rotate":
            angle = random.choice(
                [
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ]
            )
            augmentations.append(cv2.rotate(roi, angle))
        elif aug_type == "noise":
            noise = np.random.normal(0, 15, roi.shape).astype(np.uint8)
            augmentations.append(cv2.add(roi, noise))
        elif aug_type == "contrast":
            alpha = random.uniform(0.8, 1.3)
            beta = random.randint(-30, 30)
            augmentations.append(cv2.convertScaleAbs(roi, alpha=alpha, beta=beta))
    return augmentations


def generate_balanced_dataset():
    print(f"Reading Cleaned ROIs from: {RAW_ROI_DIR}")

    for folder in ["genuine", "forged"]:
        img_paths = list((RAW_ROI_DIR / folder).glob("*.png"))
        num_surviving = len(img_paths)

        if num_surviving == 0:
            continue

        # Calculate exact math to hit the TARGET_CLASS_SIZE perfectly
        base_multiplier = TARGET_CLASS_SIZE // num_surviving
        remainder = TARGET_CLASS_SIZE % num_surviving

        print(f"Processing '{folder}': Found {num_surviving} clean images.")
        print(f" -> Target size is {TARGET_CLASS_SIZE}. Distributing exactly...")

        for idx, path in enumerate(img_paths):
            roi = cv2.imread(str(path))
            if roi is not None:
                # Add 1 extra variation to the first 'remainder' number of images
                current_variations = base_multiplier + (1 if idx < remainder else 0)

                augmented_rois = augment_roi(roi, num_variations=current_variations)

                for aug_idx, aug_img in enumerate(augmented_rois):
                    out_path = BALANCED_ROI_DIR / folder / f"{path.stem}_{aug_idx}.png"
                    cv2.imwrite(str(out_path), aug_img)


if __name__ == "__main__":
    generate_balanced_dataset()
