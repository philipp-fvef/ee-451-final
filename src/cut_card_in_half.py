
import os
import cv2
import numpy as np

# take a list of cards
dir = "../data/reference_images/cropped"
file_names = os.listdir(dir)
image_paths = [os.path.join(dir, name) for name in file_names]


# the image should be cropped, not only masked out
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        continue

    # cut the image in half, save as _a and _b
    height, width, _ = img.shape
    half_width = width // 2
    half_a = img[:, :half_width]
    half_b = img[:, half_width:]

    # cut the image into thirds, keeping the outer thirds and discarding the middle third
    third_width = width // 3
    first_third_c = img[:, :third_width]
    third_third_d = img[:, 2*third_width:]

    third_height = height // 3
    first_third_e = img[:third_height, :]
    third_third_f = img[2*third_height:, :]

    # Save the halves with _a and _b suffixes
    base_name = os.path.splitext(os.path.basename(path))[0]
    save_path_a = os.path.join(dir, f"{base_name}_a.jpg")
    save_path_b = os.path.join(dir, f"{base_name}_b.jpg")
    save_path_c = os.path.join(dir, f"{base_name}_c.jpg")
    save_path_d = os.path.join(dir, f"{base_name}_d.jpg")
    save_path_e = os.path.join(dir, f"{base_name}_e.jpg")
    save_path_f = os.path.join(dir, f"{base_name}_f.jpg")

    cv2.imwrite(save_path_a, half_a)
    cv2.imwrite(save_path_b, half_b)
    cv2.imwrite(save_path_c, first_third_c)
    cv2.imwrite(save_path_d, third_third_d)
    cv2.imwrite(save_path_e, first_third_e)
    cv2.imwrite(save_path_f, third_third_f)