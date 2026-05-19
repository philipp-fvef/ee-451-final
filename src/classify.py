from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import get_config_value
from src.utils import (
    compute_descriptor_from_contours,
    process_card_image,
)


def classify_card(
    cropped: np.ndarray,
    features_path: Optional[str] = None,
    save_outputs: bool = False,
    output_root: Optional[str] = None,
    apply_opening_step: Optional[bool] = None,
    include_details: bool = False,
) -> Union[Tuple[str, str, List[np.ndarray]], Tuple[str, str, List[np.ndarray], dict]]:
    

    """
    Classify a cropped card image by matching its descriptor to reference features.

    Args:
        cropped: Cropped card image as a numpy array in RGB format.
        features_path: Path to .npz file containing reference features and metadata. If None, will use path from config.
        save_outputs: Whether to save intermediate outputs from processing the card image.
        output_root: Root directory to save outputs if save_outputs is True. If None, will use path from config.
        apply_opening_step: Whether to apply morphological opening during card image processing. If None, will use value from config.
        include_details: Whether to include detailed matching information in the return value.

    Returns:
        If include_details is False: Tuple of 
            - predicted_label
            - card_colour
            - contours (list of numpy arrays)
        If include_details is True: Tuple of
            - predicted_label
            - card_colour
            - contours (list of numpy arrays)
            - details (dict with matching information and confidence metrics)   
    
    """
    if features_path is None:
        features_path = get_config_value("paths.reference_features")

    # Process the card image to get contours and card colour
    result = process_card_image(
        cropped,
        output_root=output_root,
        save_outputs=save_outputs,
        apply_opening_step=apply_opening_step,
    )

    # Load reference features and metadata
    labels, features, meta = load_reference_features(features_path)

    # compute descriptor for the input card
    descriptor = compute_descriptor_from_contours(
        result["contours"],
        num_descriptors=meta["num_descriptors"],
        num_points=meta["num_points"],
        max_symbol_contours=meta.get(
            "max_symbol_contours",
            int(get_config_value("feature_extraction.max_symbol_contours")),
        ),
    )
    descriptor = align_descriptor(descriptor, features.shape[1])

    matching = get_config_value("matching")
    top_k = int(matching["top_k"])
    vote_min_conf = float(matching["vote_min_conf"])
    vote_min_count = int(matching["vote_min_count"])

    feature_mean = meta.get("feature_mean")
    feature_std = meta.get("feature_std")

    # match the card's descriptor to the reference features to get a predicted label and confidence details
    details = _match_descriptor_to_reference(
        result["card_colour"],
        descriptor,
        labels,
        features,
        top_k=top_k,
        vote_min_conf=vote_min_conf,
        vote_min_count=vote_min_count,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    if include_details:
        return details["predicted_label"], result["card_colour"], result["contours"], details
    else:
        return details["predicted_label"], result["card_colour"], result["contours"]


def load_reference_features(features_path: str,) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Load reference features and metadata from a .npz file.

    Args:
        features_path: Path to .npz file containing reference features and metadata.
    
    Returns:
        Tuple of (labels, features, meta) where:
            - labels: List of reference labels corresponding to the features
            - features: Numpy array of reference features
            - meta: Dictionary of metadata including feature dimensions and normalization parameters    
    """


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
    """Parse a reference label into card colour and value components."""
    for suffix in ("_bottom", "_top", "_left", "_right", "_a", "_b", "_c", "_d", "_e", "_f"):
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


def _match_descriptor_to_reference(
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
    
    """
    Match a card descriptor to reference features to predict a label and compute confidence metrics.

    Args:
        card_colour: Detected colour of the card (e.g. 'r', 'g', 'b', 'y', 'k')
        descriptor: Computed descriptor for the card image
        labels: List of reference labels corresponding to the features
        features: Numpy array of reference features
        top_k: Number of top matches to consider for voting. 
        vote_min_conf: Minimum confidence threshold to trigger voting.
        vote_min_count: Minimum count of top matches required to select a voted label. 
        feature_mean: Optional mean vector for feature normalization. 
        feature_std: Optional std vector for feature normalization.
    
    Returns:
        Dictionary containing:
            - predicted_label: The final predicted label for the card
            - matched_label: The closest matching reference label
            - matched_colour: The colour component of the matched label
            - matched_value: The value component of the matched label
            - color_override: Whether the matched colour differs from the detected colour (excluding wild cards)
            - candidate_filter: The filter applied to candidate selection based on card colour
            - candidate_count: The number of candidate reference features considered in matching
            - best_distance: The distance to the closest matching reference feature
            - second_distance: The distance to the second closest matching reference feature
            - distance_ratio: The ratio of best_distance to second_distance
            - confidence: A confidence score between 0 and 1 based on distance ratio
            - top_k: List of the top_k closest reference labels and their distances

        If matches are found, this will also include:
            - matched_variant: The specific variant of the matched label (e.g. with suffix)
            - decision: Whether the final prediction was based on "top1" or "vote"
            - voted_value: The value that was selected by voting (if applicable)
            - vote_min_conf: The confidence threshold that was used to trigger voting
            - vote_min_count: The count threshold that was used to select a voted label

    """


    if top_k is None or vote_min_conf is None or vote_min_count is None:
        if top_k is None:
            top_k = int(get_config_value("matching.top_k"))
        if vote_min_conf is None:
            vote_min_conf = float(get_config_value("matching.vote_min_conf"))
        if vote_min_count is None:
            vote_min_count = int(get_config_value("matching.vote_min_count"))

    if descriptor is None or len(labels) == 0:
        return {
            "predicted_label": None,
            "matched_label": None,
            "matched_colour": None,
            "matched_value": None,
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
