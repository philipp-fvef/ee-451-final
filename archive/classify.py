import argparse
import os
from typing import List, Optional, Tuple

import numpy as np

from utils.config import get_config_value, load_config, set_global_config
from utils.process_utils import (
    align_descriptor,
    classify_descriptor,
    classify_descriptor_with_details,
    compute_descriptor_from_contours,
    load_reference_features,
    process_card_image,
)


def classify_card(
    cropped_path: str,
    features_path: str,
    save_outputs: bool = False,
    output_root: Optional[str] = None,
    apply_opening_step: Optional[bool] = None,
) -> Tuple[str, str, List[np.ndarray]]:
    card_value, card_colour, contours, _ = classify_card_with_details(
        cropped_path,
        features_path,
        save_outputs=save_outputs,
        output_root=output_root,
        apply_opening_step=apply_opening_step,
    )
    return card_value, card_colour, contours


def classify_card_with_details(
    cropped_path: str,
    features_path: str,
    save_outputs: bool = False,
    output_root: Optional[str] = None,
    apply_opening_step: Optional[bool] = None,
) -> Tuple[str, str, List[np.ndarray], dict]:
    result = process_card_image(
        cropped_path,
        output_root=output_root,
        save_outputs=save_outputs,
        apply_opening_step=apply_opening_step,
    )

    labels, features, meta = load_reference_features(features_path)
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

    details = classify_descriptor_with_details(
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

    return details["predicted_label"], result["card_colour"], result["contours"], details


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config with matching parameters",
    )
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    set_global_config(config)

    parser = argparse.ArgumentParser(
        description="Classify a cropped UNO card image.",
        parents=[pre_parser],
    )
    parser.add_argument("image", help="Path to the cropped card image")
    parser.add_argument(
        "--features",
        default=get_config_value("paths.reference_features"),
        help="Path to the reference features .npz file",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save thresholded/mask/contour outputs alongside the image",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for outputs when --save-outputs is set",
    )
    default_opening = bool(get_config_value("feature_extraction.apply_opening_step"))
    parser.add_argument(
        "--opening",
        dest="opening",
        action="store_true",
        default=default_opening,
        help="Apply opening after closing for mask cleanup",
    )
    parser.add_argument(
        "--no-opening",
        dest="opening",
        action="store_false",
        help="Disable opening after closing for mask cleanup",
    )

    args = parser.parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.features):
        raise FileNotFoundError(
            "Reference features not found. Run process_reference_images.py first."
        )

    card_value, card_colour, contours = classify_card(
        args.image,
        args.features,
        save_outputs=args.save_outputs,
        output_root=args.output_root,
        apply_opening_step=args.opening,
    )

    print(f"Card colour: {card_colour}")
    print(f"Contours detected: {len(contours)}")
    print(f"Card value: {card_value}")


if __name__ == "__main__":
    main()
