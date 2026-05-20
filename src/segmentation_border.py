import cv2
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils import apply_colour_threshold
from src.config import get_config_value


def get_config_param(key: str, default=None):
    try:
        return get_config_value(f"segmentation.{key}")
    except (KeyError, ValueError):
        return default


WHITE_SAT_MAX = get_config_param("white_sat_max", 125)
WHITE_VAL_MIN = get_config_param("white_val_min", 200)

WHITE_BORDER_WIDTH = get_config_param("white_border_width", 20)
WHITE_RATIO_THRESH = get_config_param("white_ratio_thresh_single", 0.6)
WHITE_RATIO_THRESH_TOTAL = get_config_param("white_ratio_thresh_total", 0.75)
MIN_REGION_AREA = get_config_param("min_region_area", 7000)
MIN_AREA_RATIO = get_config_param("min_area_ratio", 0.25)
MAX_AREA_RATIO = get_config_param("max_area_ratio", 0.4)

MAX_GAP = get_config_param("max_gap", 70)
MERGING_ANGLE_TOL = get_config_param("merging_angle_tol", 2)

def get_white_mask(img, plot=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = (
        (hsv[:, :, 1] < WHITE_SAT_MAX) &
        (hsv[:, :, 2] > WHITE_VAL_MIN)
    ).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)

    if plot:
        plt.figure(figsize=(12,7))
        plt.imshow(white_mask, cmap='gray')
        plt.axis("off")
        plt.title("White mask")
        plt.show()

    return white_mask


def make_rotated_rect_mask(shape, rect, expansion=0):
    center, (w, h), angle = rect
    expanded = (center, (w + 2 * expansion, h + 2 * expansion), angle)
    box = cv2.boxPoints(expanded).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return mask


def detect_colour_regions_with_white_border(
    img_bgr,
    white_mask,
    colour_mask,
    border_width=WHITE_BORDER_WIDTH,
    white_ratio_thresh=WHITE_RATIO_THRESH,
    min_area=MIN_REGION_AREA,
    min_area_ratio=MIN_AREA_RATIO,
    max_area_ratio=MAX_AREA_RATIO,
    plot=False
):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width, border_width))

    # apply morphological closing to connect nearby components and create a more complete border
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        colour_mask, connectivity=8
    )

    regions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component = (labels == i).astype(np.uint8) * 255

        # check if the region is not squiggly (which is common for black noise) by comparing the area to the bounding box area
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = w * h
        ratio = area / bbox_area if bbox_area > 0 else 0
        # print(f"Black region {i}: area={area}, bbox_area={bbox_area}, ratio={ratio}")
        if bbox_area == 0 or ratio < min_area_ratio or ratio > max_area_ratio:
            continue

        dilated = cv2.dilate(component, kernel)
        ring = cv2.subtract(dilated, component)

        ring_pixels = int(ring.sum() / 255)
        if ring_pixels == 0:
            continue

        white_pixels = int(np.sum((white_mask > 0) & (ring > 0)))
        ratio = white_pixels / ring_pixels

        if ratio <= white_ratio_thresh:
            continue
        
        # dilate the component more to ensure we capture the full border for contour detection
        component_for_rect = cv2.dilate(component, kernel)

        contours, _ = cv2.findContours(
            component_for_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)

        regions.append((rect, box, ratio))

    return regions


def detect_black_regions_with_white_border(img_bgr,
    white_mask,
    colour_mask,
    border_width=WHITE_BORDER_WIDTH,
    min_area_ratio=MIN_AREA_RATIO,
    max_area_ratio=MAX_AREA_RATIO,
    min_area=MIN_REGION_AREA,):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width, border_width))
    # apply morphological closing to connect nearby components and create a more complete border
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)

    # plot the black mask to check if it's working correctly.
    if False:
        plt.figure(figsize=(12,7))
        plt.imshow(colour_mask, cmap='gray')
        plt.axis("off")
        plt.title("Black mask after closing")
        plt.show()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        colour_mask, connectivity=8
    )

    regions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component = (labels == i).astype(np.uint8) * 255

        # check if the region is not squiggly (which is common for black noise) by comparing the area to the bounding box area
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = w * h
        ratio = area / bbox_area if bbox_area > 0 else 0
        # print(f"Black region {i}: area={area}, bbox_area={bbox_area}, ratio={ratio}")
        if bbox_area == 0 or ratio < min_area_ratio or ratio > max_area_ratio:
            continue

        component_for_rect = cv2.dilate(component, kernel)

        contours, _ = cv2.findContours(
            component_for_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)

        regions.append((rect, box, ratio))

    return regions


def rotated_iou(rect1, rect2):
    retval, intersecting_region = cv2.rotatedRectangleIntersection(rect1, rect2)

    if retval == 0 or intersecting_region is None:
        return 0.0

    inter_area = cv2.contourArea(intersecting_region)

    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def merge_rectangles(rect1, rect2):
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)

    pts = np.vstack([box1, box2]).astype(np.float32)
    merged_rect = cv2.minAreaRect(pts)

    return merged_rect


def rect_overlap_area(rect1, rect2, expansion):
    shape = (10000, 10000)
    mask1 = make_rotated_rect_mask(shape, rect1, expansion)
    mask2 = make_rotated_rect_mask(shape, rect2, expansion)

    return ((mask1 > 0) & (mask2 > 0)).sum()


def merge_overlapping_regions(regions, angle_tol=MERGING_ANGLE_TOL, max_iters=10):

    def angle_diff(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    merged = regions.copy()

    for _ in range(max_iters):

        used = [False] * len(merged)
        new_regions = []

        for i in range(len(merged)):

            if used[i]:
                continue

            rect1, box1, score1, color1 = merged[i]

            merged_rect = rect1
            merged_score = score1

            candidates = []

            for j in range(len(merged)):

                if j == i or used[j]:
                    continue

                rect2, box2, score2, color2 = merged[j]

                if color1 != color2:
                    continue

                diff = angle_diff(rect1[2], rect2[2])
                parallel = diff < angle_tol
                perpendicular = abs(diff - 90) < angle_tol

                if not (parallel or perpendicular):
                    continue

                overlap = rect_overlap_area(merged_rect, rect2, expansion=MAX_GAP // 2)

                if overlap > 0:
                    candidates.append((overlap, j))

            # sort by overlap descending
            candidates.sort(reverse=True)

            for overlap, j in candidates:

                rect2, box2, score2, color2 = merged[j]

                proposed = merge_rectangles(merged_rect, rect2)

                (_, (w, h), _) = proposed
                area = w * h

                if area > 200000:
                    continue

                merged_rect = proposed
                merged_score = np.mean([merged_score, score2])

                used[j] = True

            new_box = cv2.boxPoints(merged_rect).astype(int)
            new_regions.append((merged_rect, new_box, merged_score, color1))

            used[i] = True

        if len(new_regions) == len(merged):
            break

        merged = new_regions

    return merged


def discard_contained_regions(regions):
    to_discard = set()

    for i in range(len(regions)):
        for j in range(len(regions)):
            if i == j or i in to_discard:
                continue

            rect_i = regions[i][0]
            rect_j = regions[j][0]

            area_i = rect_i[1][0] * rect_i[1][1]
            area_j = rect_j[1][0] * rect_j[1][1]

            if area_i >= area_j:
                continue  # only check if i is the smaller

            retval, intersection = cv2.rotatedRectangleIntersection(rect_i, rect_j)

            if retval == 0 or intersection is None:
                continue

            inter_area = cv2.contourArea(intersection)

            if inter_area / area_i > 0.95:  # tolerance for floating point
                to_discard.add(i)

    return [r for idx, r in enumerate(regions) if idx not in to_discard]


def order_box_points(pts):
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)



def segmented_cards(img_rgb, config_path='config.json', return_coords=False, plot=False):
    """Detect colored card regions with white borders in full image.

    Args:
        image: path to image file or an ndarray BGR image.
        config_path: path to JSON config (for interface compatibility, not used currently).
        return_coords: if True, return list of dicts with 'rect', 'box', 'score', 'color', 'crop'.
        plot: if True, display intermediate visualizations.

    Returns:
        If return_coords=True: List of dicts with card info and coordinates.
        If return_coords=False: List of cropped card images (BGR) for backwards compatibility.
    """

    if not isinstance(img_rgb, np.ndarray):
        try:
            img = np.array(img_rgb)
        except Exception as e:
            raise TypeError(f"Image must be a numpy array or file path, got {type(img_rgb)}: {e}")

    # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    white_mask = get_white_mask(img_rgb)

    yellow_mask = apply_colour_threshold(img_rgb, color="y")
    green_mask = apply_colour_threshold(img_rgb, color="g")
    blue_mask = apply_colour_threshold(img_rgb, color="b")
    red_mask = apply_colour_threshold(img_rgb, color="r")
    black_mask = apply_colour_threshold(img_rgb, color="k")

    all_regions = []

    for color_name, mask in [
        ("yellow", yellow_mask),
        ("green", green_mask),
        ("blue", blue_mask),
        ("red", red_mask),
        ("black", black_mask),
        
    ]:
        regions = detect_colour_regions_with_white_border(
            img_rgb,
            white_mask,
            mask.astype(np.uint8) * 255
        )

        for r in regions:
            rect, box, score = r
            all_regions.append((rect, box, score, color_name))

    """ for color_name, mask in [
        ("black", black_mask),
    ]:
        regions = detect_black_regions_with_white_border(
            img_rgb,
            white_mask,
            mask.astype(np.uint8) * 255
        )

        for r in regions:
            rect, box, score = r
            all_regions.append((rect, box, score, color_name)) """

    all_regions = merge_overlapping_regions(all_regions)
    all_regions = discard_contained_regions(all_regions)

    # merged region also shoul dbe above threshold
    all_regions = [r for r in all_regions if r[2] >= WHITE_RATIO_THRESH_TOTAL]
    # print(f"Detected {len(all_regions)} card regions with white borders.")

    if plot:
        vis = img_rgb.copy()
        for i, (rect, box, score, color_name) in enumerate(all_regions):

            cv2.drawContours(image=vis, contours=[box], contourIdx=0, color=(0,255,0), thickness=5)

            cx, cy = map(int, rect[0])

            cv2.putText(
                img=vis,
                text=f"{score:.2f}",
                org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0,0,0),
                thickness=5
            )

        plt.figure(figsize=(12,7))
        plt.imshow(vis)
        plt.axis("off")
        plt.title("Colour regions surrounded by white border")
        plt.show()

    card_results = []

    for rect, box, score, color in all_regions:
        box_points = cv2.boxPoints(rect).astype(np.float32)
        box_points = order_box_points(box_points)

        width = int(np.linalg.norm(box_points[1] - box_points[0]))
        height = int(np.linalg.norm(box_points[3] - box_points[0]))

        width = max(width, 1)
        height = max(height, 1)

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box_points, dst_pts)
        warped = cv2.warpPerspective(
            img_rgb, M, (width, height),
            flags=cv2.INTER_LINEAR
        )

        if width > height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        card_results.append({
            'rect': rect,
            'box': box,
            'score': score,
            'color': color,
            'crop': warped,
            'center': rect[0]
        })
    
    if plot:
        plt.figure(figsize=(12,7))
        for i, card in enumerate(card_results):
            plt.subplot(1, len(card_results), i + 1)
            plt.imshow(card['crop'])
            plt.axis("off")
            plt.title(f"{card['color']} ({card['score']:.2f})")
        plt.suptitle("Segmented card crops")
        plt.show()

    # print(f"Detected {len(card_results)} card regions with white borders.")

    if return_coords:
        return card_results

    return [c['crop'] for c in card_results]