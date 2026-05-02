from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import get_config_value
from utils.lab_utils import rotation_invariant, scaling_invariant, translation_invariant


def contour_shape_features(
    contour: np.ndarray,
) -> Optional[np.ndarray]:
    contour = np.asarray(contour, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return None

    contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(contour_cv))
    perimeter = float(cv2.arcLength(contour_cv, True))
    _x, _y, w, h = cv2.boundingRect(contour_cv)
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


def align_descriptor(
    descriptor: Optional[np.ndarray],
    target_len: int,
) -> Optional[np.ndarray]:
    if descriptor is None:
        return None
    if descriptor.shape[0] == target_len:
        return descriptor
    if descriptor.shape[0] > target_len:
        return descriptor[:target_len]
    return np.pad(descriptor, (0, target_len - descriptor.shape[0]), mode="constant")
