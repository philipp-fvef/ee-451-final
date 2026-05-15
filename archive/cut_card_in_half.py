import os
import cv2
import numpy as np


def cut_cards_in_half(input_dir: str) -> None:
    """
    Split each card image in a directory into multiple partial images.
    Creates 6 variants per image: _a, _b (halves), _c, _d (thirds horizontal),
    _e, _f (thirds vertical).

    Args:
        input_dir: Directory containing card images
    """
    file_names = os.listdir(input_dir)
    image_paths = [os.path.join(input_dir, name) for name in file_names]

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Image not found: {path}")
            continue

        height, width, _ = img.shape

        # Cut in halves
        half_width = width // 2
        half_a = img[:, :half_width]
        half_b = img[:, half_width:]

        # Cut into thirds (keep outer thirds)
        third_width = width // 3
        first_third_c = img[:, :third_width]
        third_third_d = img[:, 2*third_width:]

        third_height = height // 3
        first_third_e = img[:third_height, :]
        third_third_f = img[2*third_height:, :]

        # Save the variants
        base_name = os.path.splitext(os.path.basename(path))[0]
        save_path_a = os.path.join(input_dir, f"{base_name}_a.jpg")
        save_path_b = os.path.join(input_dir, f"{base_name}_b.jpg")
        save_path_c = os.path.join(input_dir, f"{base_name}_c.jpg")
        save_path_d = os.path.join(input_dir, f"{base_name}_d.jpg")
        save_path_e = os.path.join(input_dir, f"{base_name}_e.jpg")
        save_path_f = os.path.join(input_dir, f"{base_name}_f.jpg")

        cv2.imwrite(save_path_a, half_a)
        cv2.imwrite(save_path_b, half_b)
        cv2.imwrite(save_path_c, first_third_c)
        cv2.imwrite(save_path_d, third_third_d)
        cv2.imwrite(save_path_e, first_third_e)
        cv2.imwrite(save_path_f, third_third_f)
