import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops

from src.config import get_config_value


def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    Return

    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # print(f"Input image shape: {M}x{N} with {C} channels")

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # Extract RGB channels
    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    return data_red, data_green, data_blue


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # use rgb2hsv function
    img_hsv = rgb2hsv(img)
    data_h = img_hsv[:, :, 0]
    data_s = img_hsv[:, :, 1]
    data_v = img_hsv[:, :, 2]

    return data_h, data_s, data_v


def apply_rgb_threshold(img, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    """
    Apply threshold to input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)

    # Apply threshold to each channel
    img_th = (
        (data_red >= r_min)
        & (data_red <= r_max)
        & (data_green >= g_min)
        & (data_green <= g_max)
        & (data_blue >= b_min)
        & (data_blue <= b_max)
    )

    return img_th


def apply_hsv_threshold(img, h_min=0.0, h_max=1.0, s_min=0.0, s_max=1.0, v_min=0.0, v_max=1.0):
    """
    Apply threshold to the input image in hsv colorspace.

    If min is bigger than max, the threshold will be applied in a circular way.
    For example, if h_min=0.8 and h_max=0.2,
    the threshold will be applied to the hue values that are
    either greater than 0.8 or smaller than 0.2.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)

    # Apply threshold to each channel, taking into account the circular nature of hue values
    if h_min < h_max:
        img_th = (
            (data_h >= h_min) & (data_h <= h_max) & (data_s >= s_min) & (data_s <= s_max) & (data_v >= v_min) & (data_v <= v_max)
        )
    else:
        img_th = (
            ((data_h >= h_min) | (data_h <= h_max)) & (data_s >= s_min) & (data_s <= s_max) & (data_v >= v_min) & (data_v <= v_max)
        )
    return img_th


def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.
    Closing is a dilation followed by an erosion.
    It can be used to close small holes in the image.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)

    closing_disk = disk(disk_size)
    img_closing = closing(img_th, footprint=closing_disk)

    return img_closing


def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.
    Opening is an erosion followed by a dilation.
    It can be used to remove small objects from the image.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    opening_disk = disk(disk_size)
    img_opening = opening(img_th, footprint=opening_disk)

    return img_opening

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    img_holes = remove_small_holes(img_th, area_threshold=size)

    return img_holes


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    img_obj = remove_small_objects(img_th, min_size=size)

    return img_obj


def find_contours(images: np.ndarray, n: int = 0):
    """
    Find the contours for the set of images

    Args
    ----
    images: np.ndarray (N, H, W) or (H, W)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    elif images.ndim != 3:
        raise ValueError("images must have shape (H,W) or (N,H,W)")

    N = images.shape[0]
    contours = []

    for i in range(N):
        img = images[i]
        # normalize to uint8 single-channel (handles bool, float [0..1], or 0..255 ints)
        if img.dtype == np.bool_:
            img_u8 = (img.astype(np.uint8) * 255)
        elif np.issubdtype(img.dtype, np.floating):
            img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)

        _, binary = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)

        # cv2.findContours has different return signatures across versions.
        # RETR_TREE keeps both outer contours and hole contours.
        cnts = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_i = cnts[0] if len(cnts) == 2 else cnts[1]

        # return all contours for this image as a list of (K,2) arrays
        contours_per_image = []
        for c in contours_i:
            pts = c.squeeze()
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            contours_per_image.append(pts)

        # if no contours found, append an empty list
        if len(contours_per_image) == 0:
            contours.append([])
        else:
            contours.append(contours_per_image)

        # if n = 0 return all contours, else return the n largest contours by area
        if n > 0 and len(contours_per_image) > n:
            contours_per_image.sort(key=cv2.contourArea, reverse=True)
            contours[i] = contours_per_image[:n]

    return contours



def translation_invariant(features):
    """
    Make input Fourier descriptors invariant to translation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to translation of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    features_inv = features.copy()

    # Translation in spatial domain only affects the DC term (k=0) in Fourier domain.
    # Remove it to make descriptors translation-invariant.
    if features_inv.ndim == 1:
        features_inv[0] = 0
    else:
        features_inv[:, 0] = 0

    return features_inv


def rotation_invariant(features):
    """
    Make input Fourier descriptors invariant to rotation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to rotation of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    features_inv = features.copy()

    features_inv = np.abs(features)


    return features_inv


def scaling_invariant(features):
    """
    Make input Fourier descriptors invariant to scaling.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to scaling of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    features_inv = features.copy().astype(np.complex128)

    features_inv = features_inv / np.abs(features_inv[:, 1:2])

    return features_inv


def compute_distance_map(pattern: np.ndarray):
    """
    Compute the distance map for the given pattern. The values of the map are computed as
    the distance to the closest pattern contour.

    Args
    ----
    pattern: np.ndarray (28, 28)
        Pattern to process

    Return
    ------
    distance_map: np.ndarray (28, 28)
        Distance map where each entry is the distance to the closest pattern contour (shortest
        distance to pattern)
    """

    # Initialize dummy values
    distance_map = np.zeros_like(pattern)


    # ------------------
    pattern_bin = (pattern > 0).astype(np.uint8)
    contours, _ = cv2.findContours(pattern_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return distance_map
    contour_points = np.vstack([cnt.squeeze(axis=1) for cnt in contours])
    h, w = pattern.shape
    pixel_points = np.array([[x, y] for y in range(h) for x in range(w)])
    dists = euclidean_distances(pixel_points, contour_points)
    min_dists = np.min(dists, axis=1)
    distance_map = min_dists.reshape(h, w).astype(np.float32)
    # ------------------

    return distance_map


def compute_distance(imgs, d_map):
    """
    Compute the distances for each image with respect to the reference pattern using the precomputed
    distance map. The final distance is the average of all distances from the image's contour points
    to the reference pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
    d_map: np.ndarray (28, 28)
        The precomputed distance map where each entry is the distance to the closest pattern contour
        (shortest distance to pattern)

    Return
    ------
    dist: np.ndarray (N, )
        Averaged distance to pattern for each input image.
    """

    # Default values
    dist = np.zeros(len(imgs))

    for i, img in enumerate(imgs):
        img_bin = (img > 0).astype(np.uint8)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            dist[i] = 0.0
            continue
        contour_points = np.vstack([cnt.squeeze(axis=1) for cnt in contours])
        xs = contour_points[:, 0]
        ys = contour_points[:, 1]
        dist[i] = np.mean(d_map[ys, xs])

    return dist


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
    img_rgb: np.ndarray,
    output_root: Optional[str] = None,
    save_outputs: bool = True,
    verb: bool = False,
    apply_opening_step: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Process a cropped card image and optionally save thresholded/mask/contour outputs.

    Args
        img_rgb: np.ndarray (H, W, 3), input card image in RGB format
        output_root: Optional[str], directory to save outputs (thresholded, mask, contours)
        save_outputs: bool, whether to save intermediate outputs
        verb: bool, whether to print verbose output paths
        apply_opening_step: Optional[bool], whether to apply opening step in mask building
            (overrides config if not None)
    """
    # Allow passing either a filesystem path or an in-memory image (ndarray)
    cropped_dir = output_root or os.getcwd()
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

    # Load image: accept path or ndarray (assumed BGR)
    img_rgb = np.asarray(img_rgb)
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Image ndarray must be HxWx3 RGB array")
    # assume RGB input 
    filename_stem = "image"


    preview_scale = float(get_config_value("image_processing.preview_scale"))

    # Get card colour before thresholding to avoid affecting color-based classification
    preview_rgb = cv2.resize(
        src=img_rgb,
        dsize=None,
        fx=preview_scale,
        fy=preview_scale,
        interpolation=cv2.INTER_AREA,
    )
    card_colour = get_card_colour(preview_rgb)

    # build mask and find contours
    img_thresholded_filled = build_card_mask(
        img_rgb,
        card_colour,
        apply_opening_step=apply_opening_step,
    )

    img_result_rgb = (img_rgb * img_thresholded_filled[..., None]).astype(np.uint8)

    if save_outputs:
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

        contours_path = os.path.join(contours_dir, f"{filename_stem}_contours.png")
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

def parse_reference_label(label: str) -> Tuple[str, str]:
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
