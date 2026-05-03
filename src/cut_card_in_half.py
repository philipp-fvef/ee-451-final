
import os
import cv2
import numpy as np

# take a list of cards
dir = "../data/bonus_images/cropped"
file_names = [f"bonus_{i:02d}.jpg" for i in range(1, 20)]
image_paths = [os.path.join(dir, name) for name in file_names]

# cut the image in half, save as _a and _b
# the image should be cropped, not only masked out
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        continue

    height, width, _ = img.shape
    half_width = width // 2
    half_a = img[:, :half_width]
    half_b = img[:, half_width:]

    # Save the halves with _a and _b suffixes
    base_name = os.path.splitext(os.path.basename(path))[0]
    save_path_a = os.path.join(dir, f"{base_name}_a.jpg")
    save_path_b = os.path.join(dir, f"{base_name}_b.jpg")
    cv2.imwrite(save_path_a, half_a)
    cv2.imwrite(save_path_b, half_b)