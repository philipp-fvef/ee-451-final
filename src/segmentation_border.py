import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# set up path to import from utils
from pathlib import Path
import sys, os

def find_repo_root(start: Path = Path.cwd(), markers=("utils", ".git", "pyproject.toml", "setup.py")) -> Path:
    p = start.resolve()
    for _ in range(10):
        for m in markers:
            if (p / m).exists():
                return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(f"Could not find project root (markers: {markers})")

repo_root = find_repo_root()
sys.path.insert(0, str(repo_root))
print("repo_root:", repo_root)

from utils.config import load_config, set_global_config
cfg_path = repo_root / "config.json"
if not cfg_path.exists():
    raise FileNotFoundError(f"Config not found: {cfg_path}")
set_global_config(load_config(str(cfg_path)))

from utils.process_utils import apply_colour_threshold


WHITE_SAT_MAX = 125
WHITE_VAL_MIN = 200

WHITE_BORDER_WIDTH = 20
WHITE_RATIO_THRESH = 0.69
MIN_REGION_AREA = 7000

MERGING_IOU_THRESH = 0.02
MERGING_ANGLE_TOL = 10


def get_white_mask(img, plot=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(src=hsv, ksize=(21, 21), sigmaX=0)

    white_mask = (
        (hsv_blur[:, :, 1] < WHITE_SAT_MAX) &
        (hsv_blur[:, :, 2] > WHITE_VAL_MIN)
    ).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)

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
):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width, border_width))
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

        dilated = cv2.dilate(component, kernel)
        ring = cv2.subtract(dilated, component)

        ring_pixels = int(ring.sum() / 255)
        if ring_pixels == 0:
            continue

        white_pixels = int(np.sum((white_mask > 0) & (ring > 0)))
        ratio = white_pixels / ring_pixels

        if ratio <= white_ratio_thresh:
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


def merge_overlapping_regions(regions,
                              iou_thresh=MERGING_IOU_THRESH,
                              angle_tol=MERGING_ANGLE_TOL,
                              max_iters=10):
    def angle_diff(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    merged = regions.copy()

    for _ in range(max_iters):
        new_regions = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            rect1, box1, score1, color1 = merged[i]
            merged_rect = rect1
            merged_score = score1

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue

                rect2, box2, score2, color2 = merged[j]

                if color1 != color2:
                    continue

                diff = angle_diff(rect1[2], rect2[2])
                parallel = diff < angle_tol
                perpendicular = abs(diff - 90) < angle_tol

                if not (parallel or perpendicular):
                    continue

                iou = rotated_iou(merged_rect, rect2)

                if iou < iou_thresh:
                    continue

                merged_rect = merge_rectangles(merged_rect, rect2)
                merged_score = max(merged_score, score2)
                used[j] = True

            new_box = cv2.boxPoints(merged_rect).astype(int)
            new_regions.append((merged_rect, new_box, merged_score, color1))
            used[i] = True

        if len(new_regions) == len(merged):
            break

        merged = new_regions

    return merged


def order_box_points(pts):
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)



def segmented_cards(image, config_path='config.json', return_coords=False, plot=False):
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
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    else:
        img = image

    if not isinstance(img, np.ndarray):
        try:
            img = np.array(img)
        except Exception as e:
            raise TypeError(f"Image must be a numpy array or file path, got {type(img)}: {e}")

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    white_mask = get_white_mask(img_blur, plot=plot)

    yellow_mask = apply_colour_threshold(img_blur, color="y")
    green_mask = apply_colour_threshold(img_blur, color="g")
    blue_mask = apply_colour_threshold(img_blur, color="b")
    red_mask = apply_colour_threshold(img_blur, color="r")
    black_mask = apply_colour_threshold(img_blur, color="k")

    all_regions = []

    for color_name, mask in [
        ("yellow", yellow_mask),
        ("green", green_mask),
        ("blue", blue_mask),
        ("red", red_mask),
        ("black", black_mask),
    ]:
        regions = detect_colour_regions_with_white_border(
            img,
            white_mask,
            mask.astype(np.uint8) * 255
        )

        for r in regions:
            rect, box, score = r
            all_regions.append((rect, box, score, color_name))

    all_regions = merge_overlapping_regions(all_regions)

    if plot:
        vis = img.copy()
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
            img, M, (width, height),
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

    # print(f"Detected {len(card_results)} card regions with white borders.")

    if return_coords:
        return card_results

    return [c['crop'] for c in card_results]


if __name__ == "__main__":
    test_image = repo_root / "data/train_images/L1000905.jpg"
    # check if file exists
    if not os.path.isfile(test_image):
        raise FileNotFoundError(f"Test image not found: {test_image}")
    img_bgr = cv2.imread(test_image)
    img_rgb = cv2.cvtColor(src=img_bgr,
                           code=cv2.COLOR_BGR2RGB)

    cards = segmented_cards(img_rgb, plot=True)
    plt.show()