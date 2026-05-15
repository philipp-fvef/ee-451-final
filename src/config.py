import json
import os
from typing import Any, Dict, Iterable, List, Optional

_GLOBAL_CONFIG: Optional[Dict[str, Any]] = None

REQUIRED_CONFIG_KEYS: List[str] = [
    "paths.reference_features",
    "paths.reference_cropped_dir",
    "paths.reference_output_dir",
    "paths.bonus_cropped_dir",
    "paths.plots_dir",
    "matching.top_k",
    "matching.vote_min_conf",
    "matching.vote_min_count",
    "feature_extraction.num_descriptors",
    "feature_extraction.num_points",
    "feature_extraction.max_symbol_contours",
    "feature_extraction.augment_halves",
    "feature_extraction.apply_opening_step",
    "image_processing.valid_ext",
    "image_processing.preview_scale",
    "image_processing.max_contours",
    "image_processing.color_percentage_threshold",
    "image_processing.mask.disk_size_black",
    "image_processing.mask.disk_size_color",
    "image_processing.mask.opening_disk_size",
    "image_processing.color_thresholds.b.h_min",
    "image_processing.color_thresholds.b.h_max",
    "image_processing.color_thresholds.b.s_min",
    "image_processing.color_thresholds.b.s_max",
    "image_processing.color_thresholds.b.v_min",
    "image_processing.color_thresholds.b.v_max",
    "image_processing.color_thresholds.g.h_min",
    "image_processing.color_thresholds.g.h_max",
    "image_processing.color_thresholds.g.s_min",
    "image_processing.color_thresholds.g.s_max",
    "image_processing.color_thresholds.g.v_min",
    "image_processing.color_thresholds.g.v_max",
    "image_processing.color_thresholds.r.h_min",
    "image_processing.color_thresholds.r.h_max",
    "image_processing.color_thresholds.r.s_min",
    "image_processing.color_thresholds.r.s_max",
    "image_processing.color_thresholds.r.v_min",
    "image_processing.color_thresholds.r.v_max",
    "image_processing.color_thresholds.y.h_min",
    "image_processing.color_thresholds.y.h_max",
    "image_processing.color_thresholds.y.s_min",
    "image_processing.color_thresholds.y.s_max",
    "image_processing.color_thresholds.y.v_min",
    "image_processing.color_thresholds.y.v_max",
    "image_processing.color_thresholds.k.r_min",
    "image_processing.color_thresholds.k.r_max",
    "image_processing.color_thresholds.k.g_min",
    "image_processing.color_thresholds.k.g_max",
    "image_processing.color_thresholds.k.b_min",
    "image_processing.color_thresholds.k.b_max",
    "feature_dimensions.shape_feature_dim",
    "feature_dimensions.struct_feature_dim",
]


def _get_nested(config: Dict[str, Any], path: str) -> Any:
    node: Any = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def validate_config(config: Dict[str, Any], required_keys: Optional[Iterable[str]] = None) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")

    required = list(required_keys) if required_keys is not None else REQUIRED_CONFIG_KEYS
    missing: List[str] = []
    for path in required:
        try:
            value = _get_nested(config, path)
        except KeyError:
            missing.append(path)
            continue
        if value is None:
            missing.append(path)

    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required config keys: {missing_list}")


def set_global_config(
    config: Dict[str, Any],
    required_keys: Optional[Iterable[str]] = None,
) -> None:
    validate_config(config, required_keys=required_keys)
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def get_global_config() -> Dict[str, Any]:
    if _GLOBAL_CONFIG is None:
        raise ValueError("Global config is not set. Call set_global_config().")
    return _GLOBAL_CONFIG


def load_config(config_path: str, required_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    validate_config(config, required_keys=required_keys)
    return config


def get_config_value(config_or_path: Any, path: Optional[str] = None) -> Any:
    if path is None:
        config = get_global_config()
        key_path = str(config_or_path)
    else:
        config = config_or_path if config_or_path is not None else get_global_config()
        key_path = path
    try:
        return _get_nested(config, key_path)
    except KeyError as exc:
        raise KeyError(f"Missing config key: {key_path}") from exc
