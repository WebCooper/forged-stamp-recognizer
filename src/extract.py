# =====================================================================
# STAMP ROI EXTRACTION PIPELINE
# This script runs our computer vision crop pipeline on the whole dataset.
# It reads raw documents, runs segmentation and localization, and saves
# cropped stamp images into outputs/raw_rois/. 
# =====================================================================

import cv2
from src.config import DATASET_ROOT, RAW_ROI_DIR
from src.segmentation import preprocess_image, segment_stamp
from src.localization import extract_roi


def run_extraction():
    # Maps the raw input folders to cleaner target output names
    folder_mapping = {"class_0_genuine": "genuine", "class_1_forged": "forged"}

    print(f"Extracting Raw ROIs to: {RAW_ROI_DIR}")

    for input_folder, output_folder in folder_mapping.items():
        # Find all PNG files in the folder (including subdirectories just in case)
        img_paths = list((DATASET_ROOT / input_folder).rglob("*.png"))
        print(f"Processing {input_folder}: {len(img_paths)} images found.")

        success_count = 0
        for path in img_paths:
            try:
                # 1. Read document image
                img = preprocess_image(path)
                # 2. Threshold HSV values and clean with morphology to isolate stamps
                mask = segment_stamp(img)
                # 3. Score contours and crop out the best circular candidate region
                roi = extract_roi(img, mask)

                # If we successfully found and cropped a stamp
                if roi is not None:
                    out_path = RAW_ROI_DIR / output_folder / f"{path.stem}_raw.png"
                    cv2.imwrite(str(out_path), roi)
                    success_count += 1
            except Exception as e:
                # If an image fails to load or process, skip it.
                # A try-catch here is super important so one corrupt file doesn't crash 
                # our whole overnight pipeline run.
                pass

        print(f" -> Successfully extracted {success_count} ROIs for {output_folder}.")


if __name__ == "__main__":
    run_extraction()

