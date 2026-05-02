import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib.pyplot as plt

from utils.config import get_config_value
from utils.lab_utils import (
    apply_closing,
    apply_hsv_threshold,
    apply_rgb_threshold,
    apply_opening,
    find_contours,
    rotation_invariant,
    scaling_invariant,
    translation_invariant,
)

def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_colour_threshold(
    img_rgb: np.ndarray,
    color: str,
) -> np.ndarray:
    """Apply a threshold in HSV space to extract a card of a specific color."""
    thresholds = get_config_value(f"image_processing.color_thresholds.{color}")
    if color == "k":
        img_th = apply_rgb_threshold(
            img_rgb,
            r_min=int(thresholds["r_min"]),
            r_max=int(thresholds["r_max"]),
            g_min=int(thresholds["g_min"]),
            g_max=int(thresholds["g_max"]),
            b_min=int(thresholds["b_min"]),
            b_max=int(thresholds["b_max"]),
        )
    elif color in ("r", "g", "b", "y"):
        img_th = apply_hsv_threshold(
            img_rgb,
            h_min=float(thresholds["h_min"]),
            h_max=float(thresholds["h_max"]),
            s_min=float(thresholds["s_min"]),
            s_max=float(thresholds["s_max"]),
            v_min=float(thresholds["v_min"]),
            v_max=float(thresholds["v_max"]),
        )
    else:
        raise ValueError("Color must be one of 'r', 'g', 'b', 'y', or 'k'")
    return img_th


def get_card_colour(
    img_rgb: np.ndarray,
    plot: bool = False,
) -> str:
    """
    Classify the color of a card based on its image.
    Returns one of: y, r, g, b, or k (black).
    """
    

    yellow_mask = apply_colour_threshold(img_rgb, color="y")
    green_mask = apply_colour_threshold(img_rgb, color="g")
    blue_mask = apply_colour_threshold(img_rgb, color="b")
    red_mask = apply_colour_threshold(img_rgb, color="r")

    total_mask_pixels = np.sum(yellow_mask) + np.sum(green_mask) + np.sum(blue_mask) + np.sum(red_mask)
    mask_percentages = [np.sum(mask) / total_mask_pixels if total_mask_pixels > 0 else 0 
                        for mask in [yellow_mask, green_mask, blue_mask, red_mask]]

    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(yellow_mask, cmap='gray')
        axs[0].set_title("Yellow Mask")
        axs[0].set_xlabel(f"Pixels: {np.sum(yellow_mask)} ({mask_percentages[0]:.2%})")

        axs[1].imshow(green_mask, cmap='gray')
        axs[1].set_title("Green Mask")
        axs[1].set_xlabel(f"Pixels: {np.sum(green_mask)} ({mask_percentages[1]:.2%})")

        axs[2].imshow(blue_mask, cmap='gray')
        axs[2].set_title("Blue Mask")
        axs[2].set_xlabel(f"Pixels: {np.sum(blue_mask)} ({mask_percentages[2]:.2%})")

        axs[3].imshow(red_mask, cmap='gray')
        axs[3].set_title("Red Mask")
        axs[3].set_xlabel(f"Pixels: {np.sum(red_mask)} ({mask_percentages[3]:.2%})")

        plt.show()

    percentage_threshold = float(
        get_config_value("image_processing.color_percentage_threshold")
    )
    masks = [yellow_mask, green_mask, blue_mask, red_mask]
    colors = ["y", "g", "b", "r"]
    card_color = "k"
    for mask, color in zip(masks, colors):
        if np.sum(mask) > percentage_threshold * total_mask_pixels:
            card_color = color
            break
    return card_color


def build_card_mask(
    img_rgb: np.ndarray,
    card_colour: str,
    apply_opening_step: Optional[bool] = None,
) -> np.ndarray:
    mask_cfg = get_config_value("image_processing.mask")
    if apply_opening_step is None:
        apply_opening_step = bool(
            get_config_value("feature_extraction.apply_opening_step")
        )

    img_thresholded = apply_colour_threshold(img_rgb, color=card_colour)
    disk_size = (
        float(mask_cfg["disk_size_black"])
        if card_colour == "k"
        else float(mask_cfg["disk_size_color"])
    )
    img_thresholded_filled = apply_closing(img_thresholded, disk_size=disk_size)
    if apply_opening_step:
        img_thresholded_filled = apply_opening(
            img_thresholded_filled,
            disk_size=float(mask_cfg["opening_disk_size"]),
        )
    return img_thresholded_filled


def find_contours_in_image(
    img_thresholded: np.ndarray,
    max_contours: Optional[int] = None,
) -> List[np.ndarray]:
    if max_contours is None:
        max_contours = int(get_config_value("image_processing.max_contours"))
    contours = find_contours(img_thresholded, n=max_contours)[0]
    return contours


def process_card_image(
    cropped_path: str,
    output_root: Optional[str] = None,
    save_outputs: bool = True,
    verb: bool = False,
    apply_opening_step: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Process a cropped card image and optionally save thresholded/mask/contour outputs.
    """
    cropped_dir = os.path.dirname(cropped_path)
    parent_dir = os.path.dirname(cropped_dir)
    if output_root is None:
        output_root = parent_dir

    threshold_dir = os.path.join(output_root, "thresholded")
    mask_dir = os.path.join(output_root, "mask")
    contours_dir = os.path.join(output_root, "contours")

    if save_outputs:
        os.makedirs(threshold_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(contours_dir, exist_ok=True)

    img_rgb = load_image_rgb(cropped_path)

    preview_scale = float(get_config_value("image_processing.preview_scale"))
    preview_rgb = cv2.resize(
        img_rgb,
        None,
        fx=preview_scale,
        fy=preview_scale,
        interpolation=cv2.INTER_AREA,
    )
    card_colour = get_card_colour(preview_rgb)

    img_thresholded_filled = build_card_mask(
        img_rgb,
        card_colour,
        apply_opening_step=apply_opening_step,
    )

    img_result_rgb = (img_rgb * img_thresholded_filled[..., None]).astype(np.uint8)

    if save_outputs:
        filename_stem = os.path.splitext(os.path.basename(cropped_path))[0]

        threshold_path = os.path.join(threshold_dir, f"{filename_stem}_th.png")
        if verb:
            print("\tSaving thresholded image to:", threshold_path)
        cv2.imwrite(threshold_path, cv2.cvtColor(img_result_rgb, cv2.COLOR_RGB2BGR))

        mask_path = os.path.join(mask_dir, f"{filename_stem}_mask.png")
        if verb:
            print("\tSaving mask to:", mask_path)
        cv2.imwrite(mask_path, img_thresholded_filled.astype(np.uint8) * 255)

    contours = find_contours_in_image(img_thresholded_filled)

    if save_outputs:
        contour_img = np.zeros_like(img_thresholded_filled, dtype=np.uint8)
        opencv_contours = []
        for contour in contours:
            contour = np.asarray(contour).reshape(-1, 1, 2).astype(np.int32)
            opencv_contours.append(contour)
        cv2.drawContours(contour_img, opencv_contours, -1, 255, 2)

        contours_path = os.path.join(
            contours_dir,
            f"{os.path.splitext(os.path.basename(cropped_path))[0]}_contours.png",
        )
        if verb:
            print("\tSaving contours to:", contours_path)
        cv2.imwrite(contours_path, contour_img)

    return {
        "card_colour": card_colour,
        "mask": img_thresholded_filled,
        "contours": contours,
        "img_rgb": img_rgb,
    }


def contour_shape_features(
    contour: np.ndarray,
) -> Optional[np.ndarray]:
    contour = np.asarray(contour, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return None

    contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(contour_cv))
    perimeter = float(cv2.arcLength(contour_cv, True))
    x, y, w, h = cv2.boundingRect(contour_cv)
    rect_area = float(w * h) if w > 0 and h > 0 else 1.0
    extent = area / rect_area

    hull = cv2.convexHull(contour_cv)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0

    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    moments = cv2.moments(contour_cv)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    features = np.array(
        [aspect_ratio, extent, solidity, circularity, *log_hu.tolist()],
        dtype=np.float32,
    )
    shape_dim = int(get_config_value("feature_dimensions.shape_feature_dim"))
    if features.shape[0] != shape_dim:
        raise ValueError(
            "shape_feature_dim in config does not match contour_shape_features output."
        )
    return features


def contour_structural_features(
    contours: List[np.ndarray],
) -> np.ndarray:
    struct_dim = int(get_config_value("feature_dimensions.struct_feature_dim"))

    if not contours:
        return np.zeros(struct_dim, dtype=np.float32)

    areas = []
    perimeters = []
    centroids = []
    all_points = []

    for contour in contours:
        contour_cv = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
        area = float(cv2.contourArea(contour_cv))
        perimeter = float(cv2.arcLength(contour_cv, True))
        if area <= 0:
            continue
        areas.append(area)
        perimeters.append(perimeter)

        moments = cv2.moments(contour_cv)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((cx, cy))

        all_points.append(contour_cv.reshape(-1, 2))

    if not areas:
        return np.zeros(struct_dim, dtype=np.float32)

    areas_sorted = sorted(areas, reverse=True)
    area_sum = float(np.sum(areas_sorted))
    area_mean = float(np.mean(areas_sorted))
    area_std = float(np.std(areas_sorted))
    area_cv = area_std / area_mean if area_mean > 0 else 0.0

    ratio1 = areas_sorted[0] / area_sum if area_sum > 0 else 0.0
    ratio2 = areas_sorted[1] / area_sum if len(areas_sorted) > 1 and area_sum > 0 else 0.0
    ratio3 = areas_sorted[2] / area_sum if len(areas_sorted) > 2 and area_sum > 0 else 0.0

    perimeters = np.array(perimeters, dtype=np.float32)
    max_perimeter = float(perimeters.max()) if perimeters.size else 0.0
    mean_perimeter_norm = float(perimeters.mean() / max_perimeter) if max_perimeter > 0 else 0.0

    if centroids:
        centroids_arr = np.array(centroids, dtype=np.float32)
        cx_std = float(np.std(centroids_arr[:, 0]))
        cy_std = float(np.std(centroids_arr[:, 1]))
        cx_range = float(np.max(centroids_arr[:, 0]) - np.min(centroids_arr[:, 0]))
        cy_range = float(np.max(centroids_arr[:, 1]) - np.min(centroids_arr[:, 1]))
        spread_x = cx_std / (cx_range + 1e-6)
        spread_y = cy_std / (cy_range + 1e-6)
    else:
        spread_x = 0.0
        spread_y = 0.0

    if all_points:
        points = np.vstack(all_points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        box_area = float(max(x_max - x_min, 1.0) * max(y_max - y_min, 1.0))
        coverage = area_sum / box_area if box_area > 0 else 0.0
    else:
        coverage = 0.0

    features = np.array(
        [
            float(len(areas_sorted)),
            ratio1,
            ratio2,
            ratio3,
            area_cv,
            mean_perimeter_norm,
            spread_x,
            spread_y,
            coverage,
        ],
        dtype=np.float32,
    )
    if features.shape[0] != struct_dim:
        raise ValueError(
            "struct_feature_dim in config does not match contour_structural_features output."
        )
    return features


def resample_contour(
    contour: np.ndarray,
    num_points: Optional[int] = None,
) -> Optional[np.ndarray]:
    if num_points is None:
        num_points = int(get_config_value("feature_extraction.num_points"))
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return None

    pts = np.vstack([contour, contour[0]])
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    total_length = float(np.sum(seg_lengths))
    if total_length == 0:
        return None

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    sample_distances = np.linspace(0.0, total_length, num_points, endpoint=False)

    resampled = []
    for d in sample_distances:
        idx = int(np.searchsorted(cumulative, d, side="right") - 1)
        idx = min(idx, len(contour) - 1)
        seg_len = seg_lengths[idx]
        if seg_len == 0:
            resampled.append(contour[idx])
        else:
            t = (d - cumulative[idx]) / seg_len
            next_idx = (idx + 1) % len(contour)
            resampled.append(contour[idx] + t * (contour[next_idx] - contour[idx]))

    return np.asarray(resampled, dtype=np.float64)


def contour_to_fourier_descriptor(
    contour: np.ndarray,
    num_descriptors: Optional[int] = None,
    num_points: Optional[int] = None,
) -> Optional[np.ndarray]:
    if num_descriptors is None or num_points is None:
        if num_descriptors is None:
            num_descriptors = int(get_config_value("feature_extraction.num_descriptors"))
        if num_points is None:
            num_points = int(get_config_value("feature_extraction.num_points"))

    resampled = resample_contour(contour, num_points=num_points)
    if resampled is None:
        return None

    z = resampled[:, 0] + 1j * resampled[:, 1]
    fft = np.fft.fft(z)
    features = fft[:num_descriptors][np.newaxis, :]

    features = translation_invariant(features)
    if np.abs(features[0, 1]) > 1e-8:
        features = scaling_invariant(features)
    features = rotation_invariant(features)

    features = np.abs(features[0]).astype(np.float32)
    return features


def compute_descriptor_from_contours(
    contours: List[np.ndarray],
    num_descriptors: Optional[int] = None,
    num_points: Optional[int] = None,
    max_symbol_contours: Optional[int] = None,
) -> Optional[np.ndarray]:
    if num_descriptors is None or num_points is None or max_symbol_contours is None:
        if num_descriptors is None:
            num_descriptors = int(get_config_value("feature_extraction.num_descriptors"))
        if num_points is None:
            num_points = int(get_config_value("feature_extraction.num_points"))
        if max_symbol_contours is None:
            max_symbol_contours = int(
                get_config_value("feature_extraction.max_symbol_contours")
            )

    if not contours:
        return None

    areas = [
        (idx, cv2.contourArea(np.asarray(contour, dtype=np.float32)))
        for idx, contour in enumerate(contours)
    ]
    areas_sorted = sorted(areas, key=lambda item: item[1], reverse=True)

    max_symbol_contours = max(1, max_symbol_contours)
    selected = [contours[idx] for idx, _ in areas_sorted[:max_symbol_contours]]
    if not selected:
        return None

    fourier_features = []
    shape_features = []
    for contour in selected:
        descriptor = contour_to_fourier_descriptor(
            contour,
            num_descriptors=num_descriptors,
            num_points=num_points,
        )
        if descriptor is not None:
            fourier_features.append(descriptor)

        shape = contour_shape_features(contour)
        if shape is not None:
            shape_features.append(shape)

    if not fourier_features:
        return None

    fourier_mean = np.mean(np.vstack(fourier_features), axis=0)

    if shape_features:
        shape_arr = np.vstack(shape_features)
        shape_mean = shape_arr.mean(axis=0)
        shape_std = shape_arr.std(axis=0)
        shape_agg = np.concatenate([shape_mean, shape_std]).astype(np.float32)
        combined = np.concatenate([fourier_mean.astype(np.float32), shape_agg])
    else:
        combined = fourier_mean.astype(np.float32)

    structural = contour_structural_features(selected)
    combined = np.concatenate([combined, structural])

    return combined


def compute_reference_features(
    cropped_dir: str,
    features_path: str,
    output_root: Optional[str] = None,
    num_descriptors: Optional[int] = None,
    num_points: Optional[int] = None,
    max_symbol_contours: Optional[int] = None,
    apply_opening_step: Optional[bool] = None,
    augment_halves: Optional[bool] = None,
) -> Tuple[List[str], np.ndarray]:
    valid_ext = tuple(get_config_value("image_processing.valid_ext"))
    if num_descriptors is None:
        num_descriptors = int(get_config_value("feature_extraction.num_descriptors"))
    if num_points is None:
        num_points = int(get_config_value("feature_extraction.num_points"))
    if max_symbol_contours is None:
        max_symbol_contours = int(
            get_config_value("feature_extraction.max_symbol_contours")
        )
    if apply_opening_step is None:
        apply_opening_step = bool(
            get_config_value("feature_extraction.apply_opening_step")
        )
    if augment_halves is None:
        augment_halves = bool(get_config_value("feature_extraction.augment_halves"))

    filenames = sorted(
        f
        for f in os.listdir(cropped_dir)
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

        def _strip_variant_suffix(label: str) -> str:
            for suffix in ["_a", "_b", "_top", "_bottom", "_left", "_right"]:
                if label.endswith(suffix):
                    return label[:-len(suffix)]
            return label

        if augment_halves and base_label == _strip_variant_suffix(base_label):
            img_rgb = result["img_rgb"]
            height, width = img_rgb.shape[:2]
            half_width = max(1, width // 2)
            half_height = max(1, height // 2)
            halves = [
                ("a", img_rgb[:, :half_width]),
                ("b", img_rgb[:, half_width:]),
                ("top", img_rgb[:half_height, :]),
                ("bottom", img_rgb[half_height:, :]),
            ]

            for suffix, half_img in halves:
                if half_img.size == 0:
                    continue
                preview_scale = float(get_config_value("image_processing.preview_scale"))
                preview_rgb = cv2.resize(
                    half_img,
                    None,
                    fx=preview_scale,
                    fy=preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
                card_colour = get_card_colour(preview_rgb)

                half_thresholded = build_card_mask(
                    half_img,
                    card_colour,
                    apply_opening_step=apply_opening_step,
                )
                half_contours = find_contours_in_image(half_thresholded)
                half_descriptor = compute_descriptor_from_contours(
                    half_contours,
                    num_descriptors=num_descriptors,
                    num_points=num_points,
                    max_symbol_contours=max_symbol_contours,
                )
                if half_descriptor is None:
                    continue
                labels.append(f"{base_label}_{suffix}")
                features.append(half_descriptor)

    shape_dim = int(get_config_value("feature_dimensions.shape_feature_dim"))
    struct_dim = int(get_config_value("feature_dimensions.struct_feature_dim"))
    agg_shape_dim = shape_dim * 2
    feature_dim = num_descriptors + agg_shape_dim + struct_dim
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

    return labels, feature_array


def load_reference_features(
    features_path: str,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    data = np.load(features_path, allow_pickle=True)
    labels = data["labels"].tolist()
    features = data["features"].astype(np.float32)
    shape_dim_cfg = int(get_config_value("feature_dimensions.shape_feature_dim"))
    struct_dim_cfg = int(get_config_value("feature_dimensions.struct_feature_dim"))
    agg_shape_dim = shape_dim_cfg * 2
    num_descriptors = (
        int(data["num_descriptors"])
        if "num_descriptors" in data
        else int(get_config_value("feature_extraction.num_descriptors"))
    )
    num_points = (
        int(data["num_points"])
        if "num_points" in data
        else int(get_config_value("feature_extraction.num_points"))
    )
    max_symbol_contours = (
        int(data["max_symbol_contours"])
        if "max_symbol_contours" in data
        else int(get_config_value("feature_extraction.max_symbol_contours"))
    )
    shape_feature_dim = (
        int(data["shape_feature_dim"]) if "shape_feature_dim" in data else shape_dim_cfg
    )
    struct_feature_dim = (
        int(data["struct_feature_dim"]) if "struct_feature_dim" in data else struct_dim_cfg
    )
    if shape_feature_dim != shape_dim_cfg:
        raise ValueError("Config shape_feature_dim does not match features file.")
    if struct_feature_dim != struct_dim_cfg:
        raise ValueError("Config struct_feature_dim does not match features file.")
    feature_dim = (
        int(data["feature_dim"])
        if "feature_dim" in data
        else num_descriptors + agg_shape_dim + struct_feature_dim
    )

    if "feature_mean" in data and "feature_std" in data:
        feature_mean = data["feature_mean"].astype(np.float32)
        feature_std = data["feature_std"].astype(np.float32)
    else:
        if features.size == 0:
            feature_mean = np.zeros(feature_dim, dtype=np.float32)
            feature_std = np.ones(feature_dim, dtype=np.float32)
        else:
            feature_mean = features.mean(axis=0).astype(np.float32)
            feature_std = features.std(axis=0).astype(np.float32)

    return labels, features, {
        "num_descriptors": num_descriptors,
        "num_points": num_points,
        "max_symbol_contours": max_symbol_contours,
        "shape_feature_dim": shape_feature_dim,
        "struct_feature_dim": struct_feature_dim,
        "feature_dim": feature_dim,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def parse_reference_label(label: str) -> Tuple[str, str]:
    for suffix in ("_bottom", "_top", "_left", "_right", "_a", "_b"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    if label in ("wild", "draw_4"):
        return "k", label
    parts = label.split("_", 1)
    if len(parts) == 2 and parts[0] in ("r", "g", "b", "y"):
        return parts[0], parts[1]
    return "k", label


def align_descriptor(descriptor: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
    if descriptor is None:
        return None
    if descriptor.shape[0] == target_len:
        return descriptor
    if descriptor.shape[0] > target_len:
        return descriptor[:target_len]
    return np.pad(descriptor, (0, target_len - descriptor.shape[0]), mode="constant")


def classify_descriptor_with_details(
    card_colour: str,
    descriptor: Optional[np.ndarray],
    labels: List[str],
    features: np.ndarray,
    top_k: Optional[int] = None,
    vote_min_conf: Optional[float] = None,
    vote_min_count: Optional[int] = None,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if top_k is None or vote_min_conf is None or vote_min_count is None:
        if top_k is None:
            top_k = int(get_config_value("matching.top_k"))
        if vote_min_conf is None:
            vote_min_conf = float(get_config_value("matching.vote_min_conf"))
        if vote_min_count is None:
            vote_min_count = int(get_config_value("matching.vote_min_count"))

    if descriptor is None or len(labels) == 0:
        return {
            "predicted_label": "?",
            "matched_label": "?",
            "matched_colour": "?",
            "matched_value": "?",
            "color_override": False,
            "candidate_filter": "none",
            "candidate_count": 0,
            "best_distance": float("inf"),
            "second_distance": float("inf"),
            "distance_ratio": 1.0,
            "confidence": 0.0,
            "top_k": [],
        }

    if (
        descriptor is None
        or feature_mean is None
        or feature_std is None
        or features.size == 0
        or descriptor.shape[0] != feature_mean.shape[0]
        or features.shape[1] != feature_mean.shape[0]
    ):
        descriptor_scaled = descriptor
        features_scaled = features
    else:
        std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
        descriptor_scaled = (descriptor - feature_mean) / std
        features_scaled = (features - feature_mean) / std

    candidate_indices = list(range(len(labels)))
    candidate_filter = "all"

    if card_colour == "k":
        candidate_indices = [
            idx
            for idx, label in enumerate(labels)
            if parse_reference_label(label)[1] in ("wild", "draw_4")
        ]
        candidate_filter = "black-only"
    else:
        candidate_indices = [
            idx
            for idx, label in enumerate(labels)
            if parse_reference_label(label)[1] not in ("wild", "draw_4")
        ]
        candidate_filter = "non-black-only"

    if not candidate_indices:
        candidate_indices = list(range(len(labels)))
        candidate_filter = "all"
    candidate_features = features_scaled[candidate_indices]
    distances = np.linalg.norm(candidate_features - descriptor_scaled, axis=1)
    best_per_label: Dict[str, Tuple[float, str]] = {}
    for local_idx, global_idx in enumerate(candidate_indices):
        label = labels[global_idx]
        base_label = label
        for suffix in ("_bottom", "_top", "_left", "_right", "_a", "_b"):
            if base_label.endswith(suffix):
                base_label = base_label[: -len(suffix)]
                break
        dist = float(distances[int(local_idx)])
        current = best_per_label.get(base_label)
        if current is None or dist < current[0]:
            best_per_label[base_label] = (dist, label)

    sorted_items = sorted(best_per_label.items(), key=lambda item: item[1][0])
    base_labels = [item[0] for item in sorted_items]
    base_distances = [item[1][0] for item in sorted_items]
    best_variants = [item[1][1] for item in sorted_items]
    if not base_labels:
        return {
            "predicted_label": "?",
            "matched_label": "?",
            "matched_colour": "?",
            "matched_value": "?",
            "color_override": False,
            "candidate_filter": candidate_filter,
            "candidate_count": 0,
            "best_distance": float("inf"),
            "second_distance": float("inf"),
            "distance_ratio": 1.0,
            "confidence": 0.0,
            "top_k": [],
        }

    best_distance = float(base_distances[0])
    if len(base_distances) > 1:
        second_distance = float(base_distances[1])
    else:
        second_distance = best_distance

    ratio = best_distance / (second_distance + 1e-6)
    confidence = max(0.0, min(1.0, 1.0 - ratio))

    matched_label = base_labels[0]
    matched_variant = best_variants[0]
    matched_colour, matched_value = parse_reference_label(matched_label)

    top_k = max(1, int(top_k))
    top_entries = list(zip(base_labels[:top_k], base_distances[:top_k]))

    predicted = matched_value if matched_value in ("wild", "draw_4") else f"{card_colour}_{matched_value}"
    color_override = matched_colour != card_colour and matched_value not in ("wild", "draw_4")

    decision = "top1"
    voted_value = None
    if confidence < vote_min_conf:
        value_counts: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        for idx, (label, _dist) in enumerate(top_entries):
            _colour, value = parse_reference_label(label)
            value_counts[value] = value_counts.get(value, 0) + 1
            if value not in first_seen:
                first_seen[value] = idx

        candidates = [value for value, count in value_counts.items() if count >= vote_min_count]
        if candidates:
            voted_value = min(candidates, key=lambda value: first_seen[value])
            predicted = voted_value if voted_value in ("wild", "draw_4") else f"{card_colour}_{voted_value}"
            decision = "vote"

    return {
        "predicted_label": predicted,
        "matched_label": matched_label,
        "matched_colour": matched_colour,
        "matched_value": matched_value,
        "color_override": color_override,
        "candidate_filter": candidate_filter,
        "candidate_count": len(base_labels),
        "best_distance": best_distance,
        "second_distance": second_distance,
        "distance_ratio": ratio,
        "confidence": confidence,
        "top_k": top_entries,
        "matched_variant": matched_variant,
        "decision": decision,
        "voted_value": voted_value,
        "vote_min_conf": vote_min_conf,
        "vote_min_count": vote_min_count,
    }


def classify_descriptor(
    card_colour: str,
    descriptor: Optional[np.ndarray],
    labels: List[str],
    features: np.ndarray,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> str:
    if descriptor is None or len(labels) == 0:
        return "?"

    if (
        descriptor is None
        or feature_mean is None
        or feature_std is None
        or features.size == 0
        or descriptor.shape[0] != feature_mean.shape[0]
        or features.shape[1] != feature_mean.shape[0]
    ):
        descriptor_scaled = descriptor
        features_scaled = features
    else:
        std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
        descriptor_scaled = (descriptor - feature_mean) / std
        features_scaled = (features - feature_mean) / std

    candidate_indices = list(range(len(labels)))

    if card_colour == "k":
        candidate_indices = [
            idx
            for idx, label in enumerate(labels)
            if parse_reference_label(label)[1] in ("wild", "draw_4")
        ]
    else:
        candidate_indices = [
            idx
            for idx, label in enumerate(labels)
            if parse_reference_label(label)[1] not in ("wild", "draw_4")
        ]

    if not candidate_indices:
        candidate_indices = list(range(len(labels)))
    candidate_features = features_scaled[candidate_indices]
    distances = np.linalg.norm(candidate_features - descriptor_scaled, axis=1)
    best_per_label: Dict[str, Tuple[float, str]] = {}
    for local_idx, global_idx in enumerate(candidate_indices):
        label = labels[global_idx]
        base_label = label
        for suffix in ("_bottom", "_top", "_left", "_right", "_a", "_b"):
            if base_label.endswith(suffix):
                base_label = base_label[: -len(suffix)]
                break
        dist = float(distances[int(local_idx)])
        current = best_per_label.get(base_label)
        if current is None or dist < current[0]:
            best_per_label[base_label] = (dist, label)

    sorted_items = sorted(best_per_label.items(), key=lambda item: item[1][0])
    base_labels = [item[0] for item in sorted_items]
    if not base_labels:
        return "?"
    _, value = parse_reference_label(base_labels[0])

    if value in ("wild", "draw_4"):
        return value
    return f"{card_colour}_{value}"
