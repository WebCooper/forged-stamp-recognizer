import cv2
import os
from src.config import RAW_ROI_DIR


def run_cleaner():
    print(f"Starting Manual Cleaning Tool in {RAW_ROI_DIR}...")
    print("CONTROLS: [y] = Keep | [n] = Delete | [q] = Quit")

    img_paths = []
    for folder in ["genuine", "forged"]:
        img_paths.extend(list((RAW_ROI_DIR / folder).glob("*.png")))

    if not img_paths:
        print("No images found to clean! Run extract.py first.")
        return

    deleted_count = 0
    kept_count = 0

    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        display_img = cv2.resize(img, (400, 400))
        cv2.putText(
            display_img,
            "Keep: Y | Delete: N",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            f"Class: {path.parent.name}",
            (10, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Stamp Verifier (Press Y or N)", display_img)

        # Force the code to wait until a VALID key is pressed
        while True:
            key = cv2.waitKey(0) & 0xFF

            # Handle both lowercase and uppercase inputs
            if key in [ord("n"), ord("N")]:
                try:
                    os.remove(path)
                    deleted_count += 1
                    print(f"Deleted: {path.name}")
                except Exception as e:
                    print(f"Error deleting file: {e}")
                break  # Break the while loop, move to the next image

            elif key in [ord("y"), ord("Y")]:
                kept_count += 1
                break  # Break the while loop, move to the next image

            elif key in [ord("q"), ord("Q")]:
                print("Cleaning paused by user.")
                cv2.destroyAllWindows()
                print(
                    f"Session summary -> Kept: {kept_count} | Deleted: {deleted_count}"
                )
                return  # Exit the entire function immediately

    cv2.destroyAllWindows()
    print("\n--- Cleaning Complete ---")
    print(f"Kept: {kept_count} | Deleted: {deleted_count}")


if __name__ == "__main__":
    run_cleaner()
