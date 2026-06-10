# =====================================================================
# HUMAN-IN-THE-LOOP CLEANING TOOL
# This script launches a simple OpenCV GUI. We use it to manually inspect
# the extracted crops (ROIs). If the crop is bad (like we cropped empty space
# or plain text instead of a stamp), we press N to delete it. If it's a good crop,
# we press Y to keep it. This ensures our dataset contains actual stamps!
# =====================================================================

import cv2
import os
from src.config import RAW_ROI_DIR


def run_cleaner():
    print(f"Starting Manual Cleaning Tool in {RAW_ROI_DIR}...")
    print("CONTROLS: [y] = Keep | [n] = Delete | [q] = Quit")

    # Glob all PNG files from both genuine and forged subfolders
    img_paths = []
    for folder in ["genuine", "forged"]:
        img_paths.extend(list((RAW_ROI_DIR / folder).glob("*.png")))

    # If the user forgot to run extract.py, tell them first!
    if not img_paths:
        print("No images found to clean! Run extract.py first.")
        return

    deleted_count = 0
    kept_count = 0

    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            # Skip if image file is corrupted or empty
            continue

        # Resize the crop to 400x400 just to make it easier to see on screen
        display_img = cv2.resize(img, (400, 400))
        
        # Put instruction overlay text on the image
        # cv2.FONT_HERSHEY_SIMPLEX is the standard OpenCV font we learned in lab
        cv2.putText(
            display_img,
            "Keep: Y | Delete: N",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0), # Green color in BGR format
            2,
        )
        # Put class label overlay text at the bottom
        cv2.putText(
            display_img,
            f"Class: {path.parent.name}",
            (10, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0), # Yellow/cyan-ish color in BGR format
            2,
        )

        # Show the window
        cv2.imshow("Stamp Verifier (Press Y or N)", display_img)

        # Force the code to wait until a VALID key is pressed
        while True:
            # cv2.waitKey(0) blocks execution indefinitely until a key is pressed.
            # We mask with 0xFF to get the clean ASCII value (handling NumLock etc.)
            key = cv2.waitKey(0) & 0xFF

            # Handle both lowercase and uppercase inputs for N (Delete)
            if key in [ord("n"), ord("N")]:
                try:
                    os.remove(path) # Physically delete from disk
                    deleted_count += 1
                    print(f"Deleted: {path.name}")
                except Exception as e:
                    print(f"Error deleting file: {e}")
                break  # Break the while loop, move to the next image

            # Handle both lowercase and uppercase inputs for Y (Keep)
            elif key in [ord("y"), ord("Y")]:
                kept_count += 1
                break  # Break the while loop, move to the next image

            # Handle Q to quit early in case we have 1000s of images and get tired
            elif key in [ord("q"), ord("Q")]:
                print("Cleaning paused by user.")
                # Always close OpenCV windows, otherwise they freeze and python hangs lol
                cv2.destroyAllWindows()
                print(
                    f"Session summary -> Kept: {kept_count} | Deleted: {deleted_count}"
                )
                return  # Exit the entire function immediately

    # Close the window after iterating through all images
    cv2.destroyAllWindows()
    print("\n--- Cleaning Complete ---")
    print(f"Kept: {kept_count} | Deleted: {deleted_count}")


if __name__ == "__main__":
    run_cleaner()

