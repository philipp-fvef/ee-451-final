import cv2
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

from src.config import load_config, set_global_config, get_config_value

config = load_config("config.json")
set_global_config(config)

WHITE_SAT_MAX = 50
WHITE_VAL_MIN = 170
OPEN_KERNEL_SIZE = 8

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 50
HOUGH_MIN_LENGTH = 15
HOUGH_MAX_GAP = 15

DEDUP_ANGLE_TOL = 2
DEDUP_DIST_TOL = 2

ANGLE_TOL = 1
RIGHT_ANGLE_TOL = 1

BORDER_MIN = 15
BORDER_MAX = 35

CARD_SHORT = 325
CARD_LONG = 510
CARD_DIM_TOL = 30
CARD_DIAG = np.sqrt(CARD_SHORT**2 + CARD_LONG**2)

CARD_W_OUT = 325
CARD_H_OUT = 510

SCORE_EDGES_N = 3
SCORE_THRESHOLD = 0.75
WARP = False

SCORE_MODE = "mean"
WEIGHT_DET = 3.0
WEIGHT_INF = 2.0

EDGE_PAIR_BONUS = 0.2
THREE_EDGE_BONUS = 0.1

EDGE_NAMES = ["top", "right", "bottom", "left"]


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def segment_angle(seg):
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def angle_diff(a1, a2):
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def segment_midpoint(seg):
    x1, y1, x2, y2 = seg
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def point_to_line_dist(pt, seg):
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    denom = np.sqrt(dx**2 + dy**2)
    if denom < 1e-6:
        return np.inf
    return abs(dy * pt[0] - dx * pt[1] + x2 * y1 - y2 * x1) / denom


def line_direction(seg):
    x1, y1, x2, y2 = seg
    d = np.array([x2 - x1, y2 - y1], dtype=float)
    return d / (np.linalg.norm(d) + 1e-9)


def midpoint_of_confirmed_edge(mid, d):
    return [mid[0], mid[1], mid[0] + d[0] * 100, mid[1] + d[1] * 100]


def line_intersection(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def midline(seg1, seg2):
    mid1 = segment_midpoint(seg1)
    mid2 = segment_midpoint(seg2)
    mid = (mid1 + mid2) / 2
    d = line_direction(seg1)
    return mid, d


def segments_overlap(seg1, seg2, min_overlap=20):
    d = line_direction(seg1)
    p1s = np.array([seg1[0], seg1[1]], dtype=float)
    p1e = np.array([seg1[2], seg1[3]], dtype=float)
    p2s = np.array([seg2[0], seg2[1]], dtype=float)
    p2e = np.array([seg2[2], seg2[3]], dtype=float)
    t1s, t1e = np.dot(p1s, d), np.dot(p1e, d)
    t2s, t2e = np.dot(p2s, d), np.dot(p2e, d)
    lo1, hi1 = min(t1s, t1e), max(t1s, t1e)
    lo2, hi2 = min(t2s, t2e), max(t2s, t2e)
    return (min(hi1, hi2) - max(lo1, lo2)) > min_overlap


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    top = np.argmin(pts[:, 1] + pts[:, 0])
    pts = np.roll(pts, -top, axis=0)
    return pts.astype(np.float32)


def score_rect(rect, edge_map, n_samples=200):
    hits = 0
    eh, ew = edge_map.shape
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        for t in np.linspace(0, 1, n_samples // 4):
            pt = p1 + t * (p2 - p1)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < ew and 0 <= y < eh and edge_map[y, x] > 0:
                hits += 1
    return hits / n_samples


def edge_labels_from_directions(ordered_pts, confirmed_directions):
    tl, tr, br, bl = ordered_pts
    edge_dirs = [
        tr - tl,
        br - tr,
        bl - br,
        tl - bl,
    ]
    edge_dirs = [d / (np.linalg.norm(d) + 1e-9) for d in edge_dirs]

    labels = []
    for ed in edge_dirs:
        ang_e = np.degrees(np.arctan2(ed[1], ed[0])) % 180
        matched = False
        for cd in confirmed_directions:
            ang_c = np.degrees(np.arctan2(cd[1], cd[0])) % 180
            if angle_diff(ang_e, ang_c) <= ANGLE_TOL:
                matched = True
                break
        labels.append("det" if matched else "inf")
    return labels


def score_white_border(
    warped_img,
    border_width=20,
    white_sat_max=WHITE_SAT_MAX,
    white_val_min=WHITE_VAL_MIN,
    n_edges=SCORE_EDGES_N,
    edge_labels=None,
    weight_det=WEIGHT_DET,
    weight_inf=WEIGHT_INF,
    mode=SCORE_MODE,
):
    h, w = warped_img.shape[:2]
    bw = max(1, min(border_width, h // 4, w // 4))

    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    white_mask = (
        (hsv[:, :, 1] < white_sat_max)
        & (hsv[:, :, 2] > white_val_min)
    ).astype(np.uint8)

    strips = [
        white_mask[:bw, :],
        white_mask[bw:h - bw, -bw:] if h > 2 * bw else white_mask[:, -bw:],
        white_mask[-bw:, :],
        white_mask[bw:h - bw, :bw] if h > 2 * bw else white_mask[:, :bw],
    ]

    if edge_labels is None:
        edge_labels = ["inf"] * 4

    weights = [weight_det if lbl == "det" else weight_inf for lbl in edge_labels]

    raw_scores = []
    for strip in strips:
        pixels = strip.flatten()
        if len(pixels) == 0:
            raw_scores.append(0.0)
            continue

        if mode == "mean":
            raw_scores.append(float(np.mean(pixels)))
        elif mode == "continuity":
            max_run = 0
            cur_run = 0
            for px in pixels:
                if px:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 0
            raw_scores.append(max_run / len(pixels))
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'mean' or 'continuity'.")

    weighted_scores = [s * w for s, w in zip(raw_scores, weights)]
    top_indices = sorted(range(4), key=lambda i: -weighted_scores[i])[:n_edges]
    selected_weights = [weights[i] for i in top_indices]
    weight_sum = sum(selected_weights)
    if weight_sum < 1e-9:
        return 0.0
    return sum(raw_scores[i] * weights[i] for i in top_indices) / weight_sum


def line_offset(seg):
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    n = np.array([-dy, dx], dtype=float)
    n /= np.linalg.norm(n) + 1e-9
    return np.dot(n, np.array([x1, y1]))


def project_param(pt, origin, d):
    return np.dot(pt - origin, d)


def merge_colinear_segments(segments, angle_tol=3, dist_tol=10, gap_tol=20):
    if len(segments) == 0:
        return segments

    angles = np.array([segment_angle(s) for s in segments])
    offsets = np.array([line_offset(s) for s in segments])

    used = np.zeros(len(segments), dtype=bool)
    merged = []

    for i in range(len(segments)):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            if angle_diff(angles[i], angles[j]) > angle_tol:
                continue
            if abs(offsets[i] - offsets[j]) > dist_tol:
                continue
            group.append(j)
            used[j] = True

        seg = segments[group[0]]
        p0 = np.array([seg[0], seg[1]], float)
        d = line_direction(seg)

        intervals = []
        for idx in group:
            s = segments[idx]
            p1 = np.array([s[0], s[1]], float)
            p2 = np.array([s[2], s[3]], float)
            t1 = project_param(p1, p0, d)
            t2 = project_param(p2, p0, d)
            intervals.append((min(t1, t2), max(t1, t2)))

        intervals.sort()
        merged_intervals = []
        cur_lo, cur_hi = intervals[0]

        for lo, hi in intervals[1:]:
            if lo <= cur_hi + gap_tol:
                cur_hi = max(cur_hi, hi)
            else:
                merged_intervals.append((cur_lo, cur_hi))
                cur_lo, cur_hi = lo, hi
        merged_intervals.append((cur_lo, cur_hi))

        for lo, hi in merged_intervals:
            p1 = p0 + lo * d
            p2 = p0 + hi * d
            merged.append([p1[0], p1[1], p2[0], p2[1]])

    return np.array(merged)


def _as_bgr_image(img_rgb):
    if isinstance(img_rgb, (str, Path)):
        img_bgr = cv2.imread(str(img_rgb))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_rgb}")
        return img_bgr

    img = np.asarray(img_rgb)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        raise TypeError(f"Expected an image array or path, got {type(img_rgb)!r}")
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



def segmented_cards(img_rgb, config_path='config.json', return_scores=False, plot=False):
    """Detect card-like rectangles and return cropped card images."""

    img_bgr = _as_bgr_image(img_rgb)

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['axes.titlesize'] = 9

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        img_hsv,
        np.array([0, 0, WHITE_VAL_MIN]),
        np.array([180, WHITE_SAT_MAX, 255]),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL_SIZE,) * 2)
    opened = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)
    edge_map = cv2.Canny(opened, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if lines is None:
        return []

    lines = lines.reshape(-1, 4)
    lines = merge_colinear_segments(lines)
    lines = lines.astype(int)
    angles = np.array([segment_angle(l) for l in lines])

    if plot:
        line_vis = img_rgb.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(line_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(line_vis)
        ax.set_title("Raw Hough segments")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    confirmed_edges = []
    for i, j in combinations(range(len(lines)), 2):
        if angle_diff(angles[i], angles[j]) > ANGLE_TOL:
            continue
        if not segments_overlap(lines[i], lines[j], min_overlap=HOUGH_MIN_LENGTH):
            continue
        mid_j = segment_midpoint(lines[j])
        dist = point_to_line_dist(mid_j, lines[i])
        if not (BORDER_MIN < dist < BORDER_MAX):
            continue
        mid, d = midline(lines[i], lines[j])
        confirmed_edges.append((mid, d, (i, j)))

    final_edges = []
    used = [False] * len(confirmed_edges)

    for i, (mid_i, d_i, src_i) in enumerate(confirmed_edges):
        if used[i]:
            continue
        used[i] = True
        group = [i]
        ang_i = np.degrees(np.arctan2(d_i[1], d_i[0])) % 180

        for j in range(i + 1, len(confirmed_edges)):
            if used[j]:
                continue
            mid_j, d_j, src_j = confirmed_edges[j]
            ang_j = np.degrees(np.arctan2(d_j[1], d_j[0])) % 180
            if abs(angle_diff(ang_i, ang_j)) > DEDUP_ANGLE_TOL:
                continue
            if np.linalg.norm(mid_i - mid_j) > DEDUP_DIST_TOL:
                continue
            used[j] = True
            group.append(j)

        mids = np.array([confirmed_edges[k][0] for k in group])
        ds = np.array([confirmed_edges[k][1] for k in group])
        avg_mid = mids.mean(axis=0)
        avg_d = ds.mean(axis=0)
        avg_d = avg_d / (np.linalg.norm(avg_d) + 1e-9)
        srcs = tuple(idx for k in group for idx in confirmed_edges[k][2])
        final_edges.append((avg_mid, avg_d, srcs))

    if plot:
        edge_vis = img_rgb.copy()
        for mid, d, _ in final_edges:
            p1 = (mid - d * 300).astype(int)
            p2 = (mid + d * 300).astype(int)
            cv2.line(edge_vis, tuple(p1), tuple(p2), (0, 255, 0), 5)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(edge_vis)
        ax.set_title(f"Final edges: {len(final_edges)}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    inferred_edges = []
    edge_pair_rectangles = []

    for i, j in combinations(range(len(final_edges)), 2):
        mid1, d1, _ = final_edges[i]
        mid2, d2, _ = final_edges[j]

        ang1 = np.degrees(np.arctan2(d1[1], d1[0])) % 180
        ang2 = np.degrees(np.arctan2(d2[1], d2[0])) % 180

        if angle_diff(ang1, ang2) <= ANGLE_TOL:
            seg = midpoint_of_confirmed_edge(mid1, d1)
            dist = point_to_line_dist(mid2, seg)

            if abs(dist - CARD_SHORT) < CARD_DIM_TOL:
                pair_type = "short"
            elif abs(dist - CARD_LONG) < CARD_DIM_TOL:
                pair_type = "long"
            else:
                continue

            for k in range(len(final_edges)):
                if k in (i, j):
                    continue

                mid3, d3, _ = final_edges[k]
                ang3 = np.degrees(np.arctan2(d3[1], d3[0])) % 180

                if abs(angle_diff(ang1, ang3) - 90) > RIGHT_ANGLE_TOL:
                    continue
                if np.linalg.norm(mid3 - mid1) > CARD_DIAG or np.linalg.norm(mid3 - mid2) > CARD_DIAG:
                    continue

                p1 = line_intersection(mid1, d1, mid3, d3)
                p2 = line_intersection(mid2, d2, mid3, d3)
                if p1 is None or p2 is None:
                    continue

                inferred_edges.append((mid3, d3, ("perp_pair",)))

            for k, l in combinations(range(len(final_edges)), 2):
                if {k, l} & {i, j}:
                    continue

                mid3, d3, _ = final_edges[k]
                mid4, d4, _ = final_edges[l]

                ang3 = np.degrees(np.arctan2(d3[1], d3[0])) % 180
                ang4 = np.degrees(np.arctan2(d4[1], d4[0])) % 180

                if angle_diff(ang3, ang4) > ANGLE_TOL:
                    continue
                if abs(angle_diff(ang1, ang3) - 90) > RIGHT_ANGLE_TOL:
                    continue

                seg2 = midpoint_of_confirmed_edge(mid3, d3)
                dist2 = point_to_line_dist(mid4, seg2)

                dims = sorted([dist, dist2])
                if abs(dims[0] - CARD_SHORT) > CARD_DIM_TOL:
                    continue
                if abs(dims[1] - CARD_LONG) > CARD_DIM_TOL:
                    continue

                corners = []
                for ea in [(mid1, d1), (mid2, d2)]:
                    for eb in [(mid3, d3), (mid4, d4)]:
                        pt = line_intersection(ea[0], ea[1], eb[0], eb[1])
                        if pt is not None:
                            corners.append(pt)

                if len(corners) == 4:
                    edge_pair_rectangles.append(np.array(corners))

    if plot and edge_pair_rectangles:
        pair_vis = img_rgb.copy()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pair_vis)

        for pts in edge_pair_rectangles:
            pts = order_points(pts)
            poly = np.vstack([pts, pts[0]])
            ax.plot(poly[:, 0], poly[:, 1], linewidth=3)

            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            ax.text(cx, cy, "4-edge", color="yellow", fontsize=10, ha="center")

        ax.set_title(f"{len(edge_pair_rectangles)} rectangles from parallel edge pairs")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    raw_corners = []
    corner_meta = []

    for i, j in combinations(range(len(final_edges)), 2):
        mid1, d1, _ = final_edges[i]
        mid2, d2, _ = final_edges[j]

        if np.linalg.norm(mid1 - mid2) > CARD_DIAG:
            continue

        ang1 = np.degrees(np.arctan2(d1[1], d1[0])) % 180
        ang2 = np.degrees(np.arctan2(d2[1], d2[0])) % 180

        if abs(angle_diff(ang1, ang2) - 90) > RIGHT_ANGLE_TOL:
            continue

        pt = line_intersection(mid1, d1, mid2, d2)
        if pt is None:
            continue

        raw_corners.append(pt)
        corner_meta.append((pt, d1, d2))

    raw_corners = np.array(raw_corners)
    corners = []
    corner_conf_dirs = []
    corner_offset = (BORDER_MIN + BORDER_MAX) / 4.0

    if len(raw_corners) > 0:
        center_est = raw_corners.mean(axis=0)
        for pt, d1, d2 in corner_meta:
            n1 = np.array([-d1[1], d1[0]], dtype=float)
            n2 = np.array([-d2[1], d2[0]], dtype=float)
            n1 /= np.linalg.norm(n1) + 1e-9
            n2 /= np.linalg.norm(n2) + 1e-9

            if np.dot(center_est - pt, n1) > 0:
                n1 = -n1
            if np.dot(center_est - pt, n2) > 0:
                n2 = -n2

            corners.append(pt + corner_offset * (n1 + n2))
            corner_conf_dirs.append((d1, d2))

    corners = np.array(corners)

    if plot:
        corner_vis = img_rgb.copy()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(corner_vis)

        for c in corners:
            ax.plot(c[0], c[1], marker='o', color='magenta')

        ax.set_title(f"Corners from perpendicular edges: {len(corners)}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    rect_hypotheses = []

    for idx1, idx2 in combinations(range(len(corners)), 2):
        c1, c2 = corners[idx1], corners[idx2]
        d = np.linalg.norm(c1 - c2)
        if abs(d - CARD_DIAG) > CARD_DIM_TOL * 2:
            continue

        centre = (c1 + c2) / 2
        v = (c2 - c1) / np.linalg.norm(c2 - c1)
        perp = np.array([-v[1], v[0]])
        conf_dirs = list(corner_conf_dirs[idx1]) + list(corner_conf_dirs[idx2])

        for half_w, half_h in [(CARD_SHORT / 2, CARD_LONG / 2), (CARD_LONG / 2, CARD_SHORT / 2)]:
            pts = np.array([
                centre + v * half_w + perp * half_h,
                centre - v * half_w + perp * half_h,
                centre - v * half_w - perp * half_h,
                centre + v * half_w - perp * half_h,
            ])
            rect_hypotheses.append((pts, "opp_corner", conf_dirs))

    for idx1, idx2 in combinations(range(len(corners)), 2):
        c1, c2 = corners[idx1], corners[idx2]
        d = np.linalg.norm(c1 - c2)

        if abs(d - CARD_SHORT) < CARD_DIM_TOL:
            other_len = CARD_LONG
        elif abs(d - CARD_LONG) < CARD_DIM_TOL:
            other_len = CARD_SHORT
        else:
            continue

        v = (c2 - c1) / np.linalg.norm(c2 - c1)
        perp = np.array([-v[1], v[0]])
        conf_dirs = list(corner_conf_dirs[idx1]) + list(corner_conf_dirs[idx2])

        for sign in [+1, -1]:
            pts = np.array([
                c1,
                c2,
                c2 + sign * perp * other_len,
                c1 + sign * perp * other_len,
            ])
            rect_hypotheses.append((pts, "adj_corner", conf_dirs))

    for i, j in combinations(range(len(final_edges)), 2):
        mid1, d1, _ = final_edges[i]
        mid2, d2, _ = final_edges[j]

        ang1 = np.degrees(np.arctan2(d1[1], d1[0])) % 180
        ang2 = np.degrees(np.arctan2(d2[1], d2[0])) % 180

        if angle_diff(ang1, ang2) > ANGLE_TOL:
            continue

        seg = midpoint_of_confirmed_edge(mid1, d1)
        dist = point_to_line_dist(mid2, seg)

        if abs(dist - CARD_SHORT) < CARD_DIM_TOL:
            width, height = CARD_LONG, CARD_SHORT
        elif abs(dist - CARD_LONG) < CARD_DIM_TOL:
            width, height = CARD_SHORT, CARD_LONG
        else:
            continue

        centre = (mid1 + mid2) / 2
        v = d1
        perp = np.array([-v[1], v[0]])
        conf_dirs = [d1, d2]

        for w_, h_ in [(width, height), (height, width)]:
            pts = np.array([
                centre + v * w_ / 2 + perp * h_ / 2,
                centre - v * w_ / 2 + perp * h_ / 2,
                centre - v * w_ / 2 - perp * h_ / 2,
                centre + v * w_ / 2 - perp * h_ / 2,
            ])
            rect_hypotheses.append((pts, "opp_edge", conf_dirs))

    if plot:
        hyp_vis = img_rgb.copy()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(hyp_vis)

        for pts, source_type, _ in rect_hypotheses:
            ax.plot(
                np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                color="magenta",
                linewidth=2,
            )

        ax.set_title(f"Rectangle hypotheses: {len(rect_hypotheses)}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    dst = np.array(
        [[0, 0], [CARD_W_OUT - 1, 0], [CARD_W_OUT - 1, CARD_H_OUT - 1], [0, CARD_H_OUT - 1]],
        dtype=np.float32,
    )

    candidates = []

    for rect_pts, source_type, conf_dirs in rect_hypotheses:
        pts = order_points(rect_pts)

        bx, by, bw_, bh_ = cv2.boundingRect(pts.astype(np.int32))
        if bw_ > bh_:
            pts = np.roll(pts, 1, axis=0)

        labels = edge_labels_from_directions(pts, conf_dirs)

        if WARP:
            M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
            aligned = cv2.warpPerspective(img_bgr, M, (CARD_W_OUT, CARD_H_OUT))
        else:
            tl, tr = pts[0], pts[1]
            dx, dy = tr[0] - tl[0], tr[1] - tl[1]
            angle = np.degrees(np.arctan2(dy, dx))
            center = tuple(np.mean(pts, axis=0))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]))
            pts_rot = cv2.transform(np.array([pts]), M)[0]
            x, y, rw, rh = cv2.boundingRect(pts_rot.astype(np.int32))
            aligned = rotated[y:y + rh, x:x + rw]

        if aligned.shape[0] == 0 or aligned.shape[1] == 0:
            continue

        score = score_white_border(
            aligned,
            border_width=20,
            white_sat_max=WHITE_SAT_MAX,
            white_val_min=WHITE_VAL_MIN,
            n_edges=SCORE_EDGES_N,
            edge_labels=labels,
            weight_det=WEIGHT_DET,
            weight_inf=WEIGHT_INF,
            mode=SCORE_MODE,
        )

        if source_type == "edge_pair":
            score += EDGE_PAIR_BONUS
        elif source_type == "three_edge":
            score += THREE_EDGE_BONUS

        if score < SCORE_THRESHOLD:
            continue

        candidates.append((pts.astype(np.int32), aligned, score, source_type, labels))

    if plot and candidates and len(candidates) < 100:
        ncols = min(len(candidates), 5)
        nrows = (len(candidates) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.8))
        axes = np.array(axes).flatten()

        for i, (_, aligned, score, source_type, labels) in enumerate(candidates):
            axes[i].imshow(bgr_to_rgb(aligned))
            axes[i].set_title(
                f"Candidate {i}\nScore: {score:.2f}, Source: {source_type}\n"
                + " ".join(f"{n}:{l[0]}" for n, l in zip(EDGE_NAMES, labels))
            )
            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    candidates = [c for c in candidates if c[2] >= SCORE_THRESHOLD]

    candidates.sort(key=lambda x: -x[2])

    kept = []
    for cand in candidates:
        centre = cand[0].mean(axis=0)
        if all(
            np.linalg.norm(centre - k[0].mean(axis=0)) > min(CARD_SHORT, CARD_LONG) * 0.5
            for k in kept
        ):
            kept.append(cand)

    if plot:
        overlay = img_rgb.copy()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(bgr_to_rgb(overlay))

        for i, (box, _, score, source_type, labels) in enumerate(kept):
            poly = np.vstack([box, box[0]])
            ax.plot(poly[:, 0], poly[:, 1], linewidth=2)

            centre = box.mean(axis=0)
            ax.text(
                centre[0],
                centre[1],
                f"{i}({score:.2f},{source_type})",
                color="green",
                fontsize=9,
                ha="center",
            )

        ax.set_title(f"Detections: {len(kept)}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        if kept:
            ncols = min(len(kept), 5)
            nrows = (len(kept) + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.8))
            axes = np.array(axes).flatten()

            for i, (_, aligned, score, source_type, labels) in enumerate(kept):
                axes[i].imshow(bgr_to_rgb(aligned))
                axes[i].set_title(
                    f"Card {i} | {score:.2f}\n{source_type}\n"
                    + " ".join(f"{n}:{l[0]}" for n, l in zip(EDGE_NAMES, labels))
                )
                axes[i].axis("off")

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

    results = [(k[1], k[2]) for k in kept]

    if return_scores:
        return results

    return [r[0] for r in results]