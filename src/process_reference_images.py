import argparse

from utils.config import get_config_value, load_config, set_global_config
from utils.process_utils import compute_reference_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference feature embeddings.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config with pipeline defaults",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_config(config)
    cropped_dir = get_config_value("paths.reference_cropped_dir")
    output_root = get_config_value("paths.reference_output_dir")
    features_path = get_config_value("paths.reference_features")

    labels, features = compute_reference_features(
        cropped_dir,
        features_path,
        output_root=output_root,
    )

    print(f"Processed {len(labels)} reference images")
    print(f"Saved features to {features_path}")
    if features.size == 0:
        print("Warning: no features were computed. Check the cropped reference images.")


if __name__ == "__main__":
    main()
