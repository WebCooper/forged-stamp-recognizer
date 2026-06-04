import cv2
from src.config import DATASET_ROOT, RAW_ROI_DIR
from src.segmentation import preprocess_image, segment_stamp
from src.localization import extract_roi


def run_extraction():
    folder_mapping = {"class_0_genuine": "genuine", "class_1_forged": "forged"}

    print(f"Extracting Raw ROIs to: {RAW_ROI_DIR}")

    for input_folder, output_folder in folder_mapping.items():
        img_paths = list((DATASET_ROOT / input_folder).rglob("*.png"))
        print(f"Processing {input_folder}: {len(img_paths)} images found.")

        success_count = 0
        for path in img_paths:
            try:
                img = preprocess_image(path)
                mask = segment_stamp(img)
                roi = extract_roi(img, mask)

                if roi is not None:
                    out_path = RAW_ROI_DIR / output_folder / f"{path.stem}_raw.png"
                    cv2.imwrite(str(out_path), roi)
                    success_count += 1
            except Exception as e:
                pass

        print(f" -> Successfully extracted {success_count} ROIs for {output_folder}.")


if __name__ == "__main__":
    run_extraction()
