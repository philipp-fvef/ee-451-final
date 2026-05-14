import json
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

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
    denom  = np.sqrt(dx**2 + dy**2)
    if denom < 1e-6:
        return np.inf
    return abs(dy * pt[0] - dx * pt[1] + x2 * y1 - y2 * x1) / denom

def line_direction(seg):
    x1, y1, x2, y2 = seg
    d = np.array([x2 - x1, y2 - y1], dtype=float)
    return d / (np.linalg.norm(d) + 1e-9)

def midpoint_of_confirmed_edge(mid, d):
    """Return a synthetic 'segment' for point_to_line_dist compatibility."""
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
    mid  = (mid1 + mid2) / 2
    d    = line_direction(seg1)
    return mid, d

def segments_overlap(seg1, seg2, min_overlap=20):
    d   = line_direction(seg1)
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
    pts    = np.array(pts, dtype=np.float32)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts    = pts[np.argsort(angles)]
    top    = np.argmin(pts[:, 1] + pts[:, 0])
    pts    = np.roll(pts, -top, axis=0)
    return pts.astype(np.float32)

def score_rect(rect, edge_map, n_samples=200):
    hits  = 0
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

def score_white_border(
    warped_img,
    border_width=20,
    white_sat_max=50,
    white_val_min=170,
    n_edges=2
):
    """
    Score how 'card-like' a crop is by checking if its outer border is white.

    Instead of scoring the full border at once, each edge is scored
    independently and the average of the top three edges is returned
    to tolerate one obscured side.

    Args:
        warped_img: BGR image (output of cv2.warpPerspective).
        border_width: Border thickness in pixels to evaluate.
        white_sat_max: Max HSV saturation for a pixel to count as white.
        white_val_min: Min HSV value for a pixel to count as white.

    Returns:
        float in [0, 1]: mean white score of the best three edges.
    """
    h, w = warped_img.shape[:2]
    bw = max(1, min(border_width, h // 4, w // 4))

    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    # Edge strips
    edges = {
        "top": hsv[:bw, :, :],
        "bottom": hsv[-bw:, :, :],
        "left": hsv[bw:h-bw, :bw, :] if h > 2 * bw else hsv[:, :bw, :],
        "right": hsv[bw:h-bw, -bw:, :] if h > 2 * bw else hsv[:, -bw:, :],
    }

    scores = []

    for edge in edges.values():
        pixels = edge.reshape(-1, 3)
        sat = pixels[:, 1]
        val = pixels[:, 2]
        white = (sat < white_sat_max) & (val > white_val_min)
        scores.append(float(np.mean(white)))

    # Take average of the top 3 edges
    scores.sort(reverse=True)
    return float(np.mean(scores[:n_edges]))


def _read_segmentation_config(config_path):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = json.load(open(cfg_path, 'r'))
    seg = cfg.get('segmentation', {})
    # defaults matching previous script values
    defaults = dict(
        white_sat_max=50,
        white_val_min=170,
        open_kernel_size=8,
        canny_low=50,
        canny_high=150,
        hough_threshold=50,
        hough_min_length=15,
        hough_max_gap=15,
        dedup_angle_tol=1,
        dedup_dist_tol=1,
        angle_tol=1,
        right_angle_tol=1,
        border_min=15,
        border_max=35,
        card_short=325,
        card_long=512,
        card_dim_tol=20,
        card_w_out=224,
        card_h_out=347,
        score_threshold=0.7,
    )
    for k, v in defaults.items():
        seg.setdefault(k, v)
    return seg


def line_offset(seg):
    """Signed distance of line from origin in normal form."""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    n = np.array([-dy, dx], dtype=float)
    n /= np.linalg.norm(n) + 1e-9
    return np.dot(n, np.array([x1, y1]))


def project_param(pt, origin, d):
    return np.dot(pt - origin, d)


def merge_colinear_segments(segments,
                           angle_tol=3,
                           dist_tol=10,
                           gap_tol=20):
    """
    Merge colinear Hough segments into maximal segments.
    """

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

        for j in range(i+1, len(segments)):
            if used[j]:
                continue

            if angle_diff(angles[i], angles[j]) > angle_tol:
                continue

            if abs(offsets[i] - offsets[j]) > dist_tol:
                continue

            group.append(j)
            used[j] = True

        # merge group
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


def segmented_cards(image, config_path='config.json', return_scores=False, plot=False):
    """Detect card-like rectangles and return list of cropped card images.

    Args:
        image: path to image file or an ndarray BGR image.
        config_path: path to JSON config containing a `segmentation` section.
        return_scores: if True, return list of (crop, score) tuples.

    Returns:
        List of cropped card images (BGR) or list of (crop, score).
    """
    seg_cfg = _read_segmentation_config(config_path)

    # load image
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

    # parameters
    WHITE_SAT_MAX    = seg_cfg['white_sat_max']
    WHITE_VAL_MIN    = seg_cfg['white_val_min']
    OPEN_KERNEL_SIZE = seg_cfg['open_kernel_size']
    CANNY_LOW        = seg_cfg['canny_low']
    CANNY_HIGH       = seg_cfg['canny_high']
    HOUGH_THRESHOLD  = seg_cfg['hough_threshold']
    HOUGH_MIN_LENGTH = seg_cfg['hough_min_length']
    HOUGH_MAX_GAP    = seg_cfg['hough_max_gap']
    DEDUP_ANGLE_TOL  = seg_cfg['dedup_angle_tol']
    DEDUP_DIST_TOL   = seg_cfg['dedup_dist_tol']
    ANGLE_TOL        = seg_cfg['angle_tol']
    RIGHT_ANGLE_TOL  = seg_cfg['right_angle_tol']
    BORDER_MIN       = seg_cfg['border_min']
    BORDER_MAX       = seg_cfg['border_max']
    CARD_SHORT       = seg_cfg['card_short']
    CARD_LONG        = seg_cfg['card_long']
    CARD_DIM_TOL     = seg_cfg['card_dim_tol']
    CARD_W_OUT       = seg_cfg['card_w_out']
    CARD_H_OUT       = seg_cfg['card_h_out']
    SCORE_THRESHOLD  = seg_cfg['score_threshold']

    h, w = img.shape[:2]

    hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask  = cv2.inRange(hsv, np.array([0, 0, WHITE_VAL_MIN]), np.array([180, WHITE_SAT_MAX, 255]))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL_SIZE,) * 2)
    opened      = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_kernel)
    edge_map    = cv2.Canny(opened, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(edge_map, rho=1, theta=np.pi/180,
                             threshold=HOUGH_THRESHOLD,
                             minLineLength=HOUGH_MIN_LENGTH,
                             maxLineGap=HOUGH_MAX_GAP)

    if lines is None:
        return []

    lines = lines.reshape(-1,4)
    lines = merge_colinear_segments(lines)
    lines = lines.astype(int)

    angles = np.array([segment_angle(l) for l in lines])

    # confirm edges
    confirmed_edges = []
    for i, j in combinations(range(len(lines)), 2):
        if angle_diff(angles[i], angles[j]) > ANGLE_TOL:
            continue
        if not segments_overlap(lines[i], lines[j], min_overlap=HOUGH_MIN_LENGTH):
            continue
        mid_j = segment_midpoint(lines[j])
        dist  = point_to_line_dist(mid_j, lines[i])
        if not (BORDER_MIN < dist < BORDER_MAX):
            continue
        mid, d = midline(lines[i], lines[j])
        confirmed_edges.append((mid, d, (i, j)))

    # dedupe confirmed edges
    final_edges = []
    used = [False] * len(confirmed_edges)
    for i, (mid_i, d_i, src_i) in enumerate(confirmed_edges):
        if used[i]:
            continue
        used[i] = True
        group = [i]
        ang_i = np.degrees(np.arctan2(d_i[1], d_i[0])) % 180
        for j in range(i+1, len(confirmed_edges)):
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

    # detect corners
    raw_corners = []
    corner_meta = []
    for i, j in combinations(range(len(final_edges)), 2):
        mid1, d1, _ = final_edges[i]
        mid2, d2, _ = final_edges[j]
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
    corners = np.array(corners)

    # build rectangle hypotheses
    rectangles = []
    CARD_DIAG = np.sqrt(CARD_SHORT**2 + CARD_LONG**2)
    for c1, c2 in combinations(corners, 2):
        d = np.linalg.norm(c1 - c2)
        if abs(d - CARD_DIAG) > CARD_DIM_TOL * 2:
            continue
        centre = (c1 + c2) / 2
        v = c2 - c1
        v /= np.linalg.norm(v)
        perp = np.array([-v[1], v[0]])
        half_w = CARD_SHORT / 2
        half_h = CARD_LONG  / 2
        pts1 = np.array([
            centre + v*half_w + perp*half_h,
            centre - v*half_w + perp*half_h,
            centre - v*half_w - perp*half_h,
            centre + v*half_w - perp*half_h,
        ])
        pts2 = np.array([
            centre + v*half_h + perp*half_w,
            centre - v*half_h + perp*half_w,
            centre - v*half_h - perp*half_w,
            centre + v*half_h - perp*half_w,
        ])
        rectangles.append(pts1)
        rectangles.append(pts2)

    # adjacent-corner hypotheses
    for c1, c2 in combinations(corners, 2):
        d = np.linalg.norm(c1 - c2)
        if abs(d - CARD_SHORT) < CARD_DIM_TOL:
            edge_len   = CARD_SHORT
            other_len  = CARD_LONG
        elif abs(d - CARD_LONG) < CARD_DIM_TOL:
            edge_len   = CARD_LONG
            other_len  = CARD_SHORT
        else:
            continue
        v = c2 - c1
        v /= np.linalg.norm(v)
        perp = np.array([-v[1], v[0]])
        pts1 = np.array([
            c1,
            c2,
            c2 + perp * other_len,
            c1 + perp * other_len,
        ])
        pts2 = np.array([
            c1,
            c2,
            c2 - perp * other_len,
            c1 - perp * other_len,
        ])
        rectangles.append(pts1)
        rectangles.append(pts2)

    # rectangles from parallel edge pairs
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
        pts = np.array([
            centre + v*width/2 + perp*height/2,
            centre - v*width/2 + perp*height/2,
            centre - v*width/2 - perp*height/2,
            centre + v*width/2 - perp*height/2
        ])
        pts_swapped = np.array([
            centre + v*height/2 + perp*width/2,
            centre - v*height/2 + perp*width/2,
            centre - v*height/2 - perp*width/2,
            centre + v*height/2 - perp*width/2,
        ])
        rectangles.append(pts)
        rectangles.append(pts_swapped)

    # validate rectangles
    dst = np.array([[0,0],[CARD_W_OUT-1,0],[CARD_W_OUT-1,CARD_H_OUT-1],[0,CARD_H_OUT-1]], dtype=np.float32)
    candidates = []
    for rect in rectangles:
        pts = order_points(rect)
        bx, by, bw_, bh_ = cv2.boundingRect(pts.astype(np.int32))
        if bw_ > bh_:
            pts = np.roll(pts, 1, axis=0)
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (CARD_W_OUT, CARD_H_OUT))
        score = score_white_border(
            warped,
            border_width=20,
            white_sat_max=WHITE_SAT_MAX,
            white_val_min=WHITE_VAL_MIN
        )
        if score < SCORE_THRESHOLD:
            continue
        candidates.append((pts.astype(np.int32), warped, float(score)))

    # deduplicate and keep best
    candidates.sort(key=lambda x: -x[2])
    kept = []
    for cand in candidates:
        centre = cand[0].mean(axis=0)
        if all(np.linalg.norm(centre - k[0].mean(axis=0)) > min(CARD_SHORT, CARD_LONG) * 0.4
               for k in kept):
            kept.append(cand)

    
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for rect, _, score in candidates:
            color = (0, 255, 0) if any(np.array_equal(rect, k[0]) for k in kept) else (255, 0, 0)
            cv2.polylines(img, [rect], isClosed=True, color=color, thickness=2)
            c = rect.mean(axis=0).astype(int)
            cv2.putText(img, f"{score:.2f}", (c[0]-20, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    results = [(k[1], k[2]) for k in kept]
    if return_scores:
        return results
    
    return [r[0] for r in results]
