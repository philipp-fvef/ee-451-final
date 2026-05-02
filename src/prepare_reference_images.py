import argparse
import os
from typing import Any, Dict

import cv2

from utils.config import get_config_value, load_config, set_global_config

REFERENCE_IMAGES = {
    "data/reference_images/L1000765.jpg": {
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
    "data/reference_images/L1000766.jpg": {
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
    "data/reference_images/L1000767.jpg": {
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
    "data/reference_images/L1000768.jpg": {
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
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for card_name, crop in crop_map.items():
            y0, y1 = crop["y"]
            x0, x1 = crop["x"]
            cropped_rgb = img_rgb[y0:y1, x0:x1]
            out_path = os.path.join(output_dir, f"{card_name}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop reference images into cards.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config with pipeline defaults",
    )
    args = parser.parse_args()

    config: Dict[str, Any] = load_config(args.config)
    set_global_config(config)
    output_dir = get_config_value("paths.reference_cropped_dir")
    crop_reference_images(output_dir)


if __name__ == "__main__":
    main()
