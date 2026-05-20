import os
from typing import Dict, Any, List

import cv2
import numpy as np

from src.config import get_config_value, load_config, set_global_config
from src.utils import (
    process_card_image,
    compute_descriptor_from_contours,
    get_card_colour,
    build_card_mask,
    find_contours_in_image,
)

REFERENCE_IMAGES = {
    "reference_images/L1000765.jpg": {
        "y_3": {"x": [1250, 1650], "y": [630, 1230]},
        "y_2": {"x": [1270, 1670], "y": [1280, 1860]},
        "y_1": {"x": [1280, 1680], "y": [1880, 2470]},
        "r_3": {"x": [1680, 2100], "y": [640, 1230]},
        "r_2": {"x": [1710, 2100], "y": [1290, 1860]},
        "r_1": {"x": [1730, 2120], "y": [1890, 2470]},
        "b_3": {"x": [2120, 2510], "y": [630, 1220]},
        "b_2": {"x": [2120, 2530], "y": [1270, 1840]},
        "b_1": {"x": [2160, 2580], "y": [1880, 2460]},
        "g_3": {"x": [2540, 2940], "y": [630, 1210]},
        "g_2": {"x": [2550, 2940], "y": [1250, 1830]},
        "g_1": {"x": [2580, 2970], "y": [1880, 2450]},
    },
    "reference_images/L1000766.jpg": {
        "y_6": {"x": [1500, 1910], "y": [630, 1230]},
        "y_5": {"x": [1550, 1960], "y": [1280, 1860]},
        "y_4": {"x": [1620, 2040], "y": [1880, 2470]},
        "r_6": {"x": [1960, 2370], "y": [630, 1230]},
        "r_5": {"x": [2000, 2400], "y": [1280, 1860]},
        "r_4": {"x": [2040, 2430], "y": [1880, 2470]},
        "b_6": {"x": [2450, 2860], "y": [620, 1210]},
        "b_5": {"x": [2470, 2900], "y": [1280, 1860]},
        "b_4": {"x": [2510, 2930], "y": [1880, 2470]},
        "g_6": {"x": [2910, 3340], "y": [580, 1180]},
        "g_5": {"x": [2950, 3390], "y": [1280, 1860]},
        "g_4": {"x": [3000, 3400], "y": [1890, 2470]},
    },
    "reference_images/L1000767.jpg": {
        "draw_4": {"x": [1140, 1550], "y": [900, 1480]},
        "wild": {"x": [1190, 1610], "y": [1770, 2350]},
        "y_7": {"x": [1850, 2300], "y": [1980, 2540]},
        "y_8": {"x": [1790, 2200], "y": [1340, 1920]},
        "y_9": {"x": [1700, 2120], "y": [680, 1280]},
        "r_7": {"x": [2300, 2750], "y": [1950, 2550]},
        "r_8": {"x": [2250, 2650], "y": [1330, 1900]},
        "r_9": {"x": [2220, 2650], "y": [620, 1220]},
        "b_7": {"x": [2750, 3170], "y": [1930, 2520]},
        "b_8": {"x": [2700, 3100], "y": [1320, 1900]},
        "b_9": {"x": [2750, 3170], "y": [650, 1290]},
        "g_7": {"x": [3250, 3650], "y": [1900, 2500]},
        "g_8": {"x": [3220, 3620], "y": [1320, 1900]},
        "g_9": {"x": [3250, 3670], "y": [600, 1200]},
    },
    "reference_images/L1000768.jpg": {
        "y_draw_2": {"x": [100, 500], "y": [1720, 2300]},
        "b_draw_2": {"x": [570, 950], "y": [1720, 2300]},
        "r_draw_2": {"x": [950, 1340], "y": [1720, 2300]},
        "g_draw_2": {"x": [1350, 1790], "y": [1680, 2300]},
        "y_reverse": {"x": [1700, 2100], "y": [320, 920]},
        "r_reverse": {"x": [2150, 2550], "y": [300, 880]},
        "b_reverse": {"x": [2630, 3070], "y": [150, 780]},
        "g_reverse": {"x": [3100, 3520], "y": [200, 820]},
        "y_skip": {"x": [1790, 2200], "y": [1050, 1620]},
        "r_skip": {"x": [2200, 2620], "y": [1000, 1600]},
        "b_skip": {"x": [2630, 3100], "y": [950, 1550]},
        "g_skip": {"x": [3180, 3620], "y": [900, 1500]},
        "y_0": {"x": [1880, 2300], "y": [1650, 2250]},
        "r_0": {"x": [2330, 2750], "y": [1630, 2230]},
        "b_0": {"x": [2790, 3200], "y": [1600, 2200]},
        "g_0": {"x": [3230, 3680], "y": [1580, 2190]},
    },
}


def crop_reference_images(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for image_path, crop_map in REFERENCE_IMAGES.items():
        image_path_full = image_path
        img_bgr = cv2.imread(image_path_full)
        if img_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {image_path_full}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for card_name, crop in crop_map.items():
            y0, y1 = crop["y"]
            x0, x1 = crop["x"]
            cropped_rgb = img_rgb[y0:y1, x0:x1]
            out_path = os.path.join(output_dir, f"{card_name}.jpg")
            # print(f"Cropping {image_path_full} to {out_path}")
            cv2.imwrite(out_path, cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))


def _strip_variant_suffix(label: str) -> str:
    for suffix in ["_a", "_b", "_c", "_d", "_e", "_f", "_top", "_bottom", "_left", "_right"]:
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def initialize_reference_images() -> None:

    """
    Initialize reference images and compute features.
    This function crops the reference images, processes them to extract features, and saves the features to a .npz file.

    Files:
    - Cropped reference images are saved to the directory specified by "paths.reference_cropped_dir" in the config.
    - Extracted features and metadata are saved to the file specified by "paths.reference_features" in the config.

    Args:
        None
    Returns:
        None
    """

    config = load_config("config.json")
    set_global_config(config)

    cropped_dir = get_config_value("paths.reference_output_dir")
    output_root = get_config_value("paths.reference_output_dir")
    features_path = get_config_value("paths.reference_features")

    valid_ext = tuple(get_config_value("image_processing.valid_ext"))
    preview_scale = float(get_config_value("image_processing.preview_scale"))
    num_descriptors = int(get_config_value("feature_extraction.num_descriptors"))
    num_points = int(get_config_value("feature_extraction.num_points"))
    max_contours = int(get_config_value("image_processing.max_contours"))
    max_symbol_contours = int(get_config_value("image_processing.max_contours"))
    apply_opening_step = bool(get_config_value("feature_extraction.apply_opening_step"))
    augment_halves = bool(get_config_value("feature_extraction.augment_halves"))
    shape_dim = int(get_config_value("feature_dimensions.shape_feature_dim"))
    struct_dim = int(get_config_value("feature_dimensions.struct_feature_dim"))

    crop_reference_images(cropped_dir)

    filenames = sorted(
        f for f in os.listdir(cropped_dir)
        if f.lower().endswith(valid_ext) and not f.lower().endswith("_th.png")
    )

    labels: List[str] = []
    features: List[np.ndarray] = []

    for filename in filenames:
        img_path = os.path.join(cropped_dir, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print(f"Processing reference image: {img_path}")
        result = process_card_image(
            img_rgb,
            output_root=output_root,
            save_outputs=True,
            apply_opening_step=apply_opening_step,
        )
        base_label = os.path.splitext(filename)[0]
        descriptor = compute_descriptor_from_contours(
            result["contours"],
            num_descriptors=num_descriptors,
            num_points=num_points,
            max_contours=max_symbol_contours,
        )
        if descriptor is not None:
            labels.append(base_label)
            features.append(descriptor)

        if augment_halves and base_label == _strip_variant_suffix(base_label):
            img_rgb = result["img_rgb"]
            height, width = img_rgb.shape[:2]
            half_width = max(1, width // 2)
            third_width = max(1, width // 3)
            third_height = max(1, height // 3)
            partials = [
                ("a", img_rgb[:, :half_width]),
                ("b", img_rgb[:, half_width:]),
                ("c", img_rgb[:, :third_width]),
                ("d", img_rgb[:, 2 * third_width:]),
                ("e", img_rgb[:third_height, :]),
                ("f", img_rgb[2 * third_height:, :]),
            ]

            for suffix, partial_img in partials:
                if partial_img.size == 0:
                    continue
                preview_rgb = cv2.resize(
                    partial_img,
                    None,
                    fx=preview_scale,
                    fy=preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
                card_colour = get_card_colour(preview_rgb)
                partial_thresholded = build_card_mask(
                    partial_img,
                    card_colour,
                    apply_opening_step=apply_opening_step,
                )
                partial_contours = find_contours_in_image(partial_thresholded)
                partial_descriptor = compute_descriptor_from_contours(
                    partial_contours,
                    num_descriptors=num_descriptors,
                    num_points=num_points,
                    max_contours=max_symbol_contours,
                )
                if partial_descriptor is None:
                    continue
                labels.append(f"{base_label}_{suffix}")
                features.append(partial_descriptor)

    feature_dim = num_descriptors + shape_dim * 2 + struct_dim
    feature_array = (
        np.vstack(features).astype(np.float32)
        if features
        else np.empty((0, feature_dim), dtype=np.float32)
    )

    if feature_array.size == 0:
        feature_mean = np.zeros(feature_dim, dtype=np.float32)
        feature_std = np.ones(feature_dim, dtype=np.float32)
    else:
        feature_mean = feature_array.mean(axis=0).astype(np.float32)
        feature_std = feature_array.std(axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    np.savez(
        features_path,
        labels=np.array(labels),
        features=feature_array,
        feature_mean=feature_mean,
        feature_std=feature_std,
        num_descriptors=np.array(num_descriptors, dtype=np.int32),
        num_points=np.array(num_points, dtype=np.int32),
        max_symbol_contours=np.array(max_symbol_contours, dtype=np.int32),
        shape_feature_dim=np.array(shape_dim, dtype=np.int32),
        struct_feature_dim=np.array(struct_dim, dtype=np.int32),
        feature_dim=np.array(feature_dim, dtype=np.int32),
    )

    print(f"Processed {len(labels)} reference images")
    print(f"Saved features to {features_path}")
    if feature_array.size == 0:
        print("Warning: no features were computed. Check the cropped reference images.")


if __name__ == "__main__":
    initialize_reference_images()
