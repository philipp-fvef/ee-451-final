import os
import numpy as np
import cv2

from typing import List, Tuple

from src.config import get_config_value, load_config, set_global_config
from src.utils import process_card_image, compute_descriptor_from_contours, get_card_colour, build_card_mask, find_contours_in_image


def _strip_variant_suffix(label: str) -> str:
    for suffix in ["_a", "_b", "_c", "_d", "_e", "_f", "_top", "_bottom", "_left", "_right"]:
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


config = load_config("config.json")
set_global_config(config)

cropped_dir = get_config_value("paths.reference_cropped_dir")
output_root = get_config_value("paths.reference_output_dir")
features_path = get_config_value("paths.reference_features")

valid_ext = tuple(get_config_value("image_processing.valid_ext"))
preview_scale = float(get_config_value("image_processing.preview_scale"))
num_descriptors = int(get_config_value("feature_extraction.num_descriptors"))
num_points = int(get_config_value("feature_extraction.num_points"))
max_symbol_contours = int(get_config_value("feature_extraction.max_symbol_contours"))
apply_opening_step = bool(get_config_value("feature_extraction.apply_opening_step"))
augment_halves = bool(get_config_value("feature_extraction.augment_halves"))
shape_dim = int(get_config_value("feature_dimensions.shape_feature_dim"))
struct_dim = int(get_config_value("feature_dimensions.struct_feature_dim"))

filenames = sorted(
    f for f in os.listdir(cropped_dir)
    if f.lower().endswith(valid_ext) and not f.lower().endswith("_th.png")
)

labels: List[str] = []
features: List[np.ndarray] = []

for filename in filenames:
    img_path = os.path.join(cropped_dir, filename)
    result = process_card_image(
        img_path,
        output_root=output_root,
        save_outputs=True,
        apply_opening_step=apply_opening_step,
    )
    base_label = os.path.splitext(filename)[0]
    descriptor = compute_descriptor_from_contours(
        result["contours"],
        num_descriptors=num_descriptors,
        num_points=num_points,
        max_symbol_contours=max_symbol_contours,
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
                max_symbol_contours=max_symbol_contours,
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