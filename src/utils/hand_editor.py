"""
hand_editor.py - Robust Analog Watch Hand Editor
==================================================
Modular CV pipeline for replacing clock hands on an analog watch image.

Public API
----------
    parse_time(hour, minute, second)
    detect_watch_circle(image)
    compute_hand_angles(hour, minute, second)
    detect_existing_hands(image, cx, cy, radius)
    remove_hands(image, mask, cx, cy, radius)
    draw_clock_hand(canvas, cx, cy, angle_deg, length, width, color)
    render_new_hands(face_image, h_ang, m_ang, s_ang, cx, cy, radius, ...)
    process_watch_pair(time_tuple, analog_image, output_path, debug_dir)

Design principles
-----------------
- Deterministic CV only (no neural nets in this module)
- Each function is independently testable and replaceable
- Correct hand-angle formulas (clockwise from 12 o'clock)
- Smudge-free inpainting: fill → TELEA edge-blend (no NS reconstruction)
- SSAA rendering (4×) for crisp anti-aliased hands
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ===========================================================================
# 1.  TIME UTILITIES
# ===========================================================================

def parse_time(hour: int, minute: int, second: int) -> Tuple[int, int, int]:
    """
    Validate and normalise a raw time tuple.

    Returns
    -------
    (h, m, s) in [0..23], [0..59], [0..59]
    """
    return int(hour) % 24, int(minute) % 60, int(second) % 60


def compute_hand_angles(
    hour: int,
    minute: int,
    second: int,
) -> Tuple[float, float, float]:
    """
    Compute hand angles in degrees, measured clockwise from 12 o'clock.

    Formulas
    --------
    second_angle = 6 * s
    minute_angle = 6 * m + 0.1 * s
    hour_angle   = 30 * (h % 12) + 0.5 * m + s / 120

    Returns
    -------
    (hour_angle, minute_angle, second_angle) - all in [0, 360)
    """
    h = hour % 12
    second_angle = 6.0 * second
    minute_angle = 6.0 * minute + 0.1 * second
    hour_angle   = 30.0 * h + 0.5 * minute + second / 120.0
    return float(hour_angle), float(minute_angle), float(second_angle)


def _angle_to_point(
    cx: int, cy: int, angle_deg: float, length: float,
) -> Tuple[int, int]:
    """
    Convert (cx, cy, angle_deg, length) to an endpoint pixel.

    Convention:  angle = 0 → 12 o'clock, increases clockwise.
    theta = radians(angle_deg − 90)
    x = cx + length * cos(theta)
    y = cy + length * sin(theta)
    """
    theta = math.radians(angle_deg - 90.0)
    x = cx + length * math.cos(theta)
    y = cy + length * math.sin(theta)
    return int(round(x)), int(round(y))


# ===========================================================================
# 2.  CIRCLE DETECTION
# ===========================================================================

def detect_watch_circle(
    image: np.ndarray,
    dp: float = 1.2,
    param1: float = 60.0,
    param2: float = 30.0,
) -> Tuple[int, int, int]:
    """
    Detect the outermost watch-face circle using HoughCircles.

    Primary detector:  cv2.HoughCircles on a lightly blurred grayscale image.
    Fallback:          contour-based circle/ellipse estimation when Hough fails.
    Last resort:       image centre / 90 % of min-half-dimension so downstream
                       code always receives a valid circle.

    Parameters
    ----------
    image  : RGB ndarray
    dp     : inverse resolution ratio for HoughCircles
    param1 : Canny high threshold
    param2 : accumulator threshold (lower → more circles)

    Returns
    -------
    (cx, cy, radius)  - all in pixels
    """
    h, w = image.shape[:2]
    rough_r = min(h, w) // 2 - 5
    cx_def, cy_def = w // 2, h // 2

    gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=rough_r,
        param1=param1,
        param2=param2,
        minRadius=int(rough_r * 0.55),
        maxRadius=int(rough_r * 1.05),
    )

    if circles is None:
        # ── Fallback: contour-based circle / ellipse estimation ─────────────
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = (cx_def, cy_def)
        best_radius = int(rough_r * 0.90)
        best_score = float("inf")

        for cnt in contours:
            if len(cnt) < 8:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            r = int(r)
            if r <= 0:
                continue

            # Require plausible radius range for a dial
            if r < rough_r * 0.40 or r > rough_r * 1.30:
                continue

            dx = x - cx_def
            dy = y - cy_def
            centre_dist2 = dx * dx + dy * dy
            area = cv2.contourArea(cnt)

            # Score: prefer large, centred contours
            score = centre_dist2 - 0.001 * area
            if score < best_score:
                best_score = score
                best_center = (int(x), int(y))
                best_radius = r

        return best_center[0], best_center[1], best_radius

    circles = np.round(circles[0]).astype(int)
    # Pick the circle whose centre is closest to the image centre
    best = min(circles, key=lambda c: (c[0] - cx_def) ** 2 + (c[1] - cy_def) ** 2)
    return int(best[0]), int(best[1]), int(best[2])


# ===========================================================================
# 3.  HAND DETECTION
# ===========================================================================

def _sample_face_color(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> np.ndarray:
    """
    Sample the dominant background colour from an inner annulus (20–40 % of
    radius) that sits away from numerals/ticks and from the border.

    Uses the modal 32-bin colour bucket for robustness against hand pixels.
    Returns float32 RGB array of shape (3,).
    """
    h, w = image.shape[:2]
    inner = max(1, int(radius * 0.20))
    outer = int(radius * 0.40)
    pixels = []
    for r in range(inner, outer + 1):
        for deg in range(0, 360, 6):
            rad = math.radians(deg - 90)
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))
            if 0 <= x < w and 0 <= y < h:
                pixels.append(image[y, x].astype(np.float32))

    if not pixels:
        return np.array([240.0, 240.0, 240.0], dtype=np.float32)

    arr = np.stack(pixels)  # (N, 3)
    quant = (arr // 32).astype(np.int32)
    # Find modal bucket
    keys, counts = np.unique(
        quant.view(np.dtype((np.void, quant.dtype.itemsize * 3))),
        return_counts=True,
    )
    best_bucket = np.frombuffer(keys[np.argmax(counts)], dtype=np.int32) * 32 + 16
    mask_row = np.all(quant == best_bucket, axis=1)
    refined = arr[mask_row].mean(axis=0) if mask_row.any() else arr.mean(axis=0)
    return refined.astype(np.float32)


def _legacy_detect_existing_hands(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    dilation: int = 7,
) -> np.ndarray:
    """
    Legacy background-difference based hand detector.

    Kept as a robust fallback when Hough-based line detection fails or yields
    no usable candidates. Returns a binary mask (uint8, 255 = hand pixel).
    """
    h, w = image.shape[:2]
    face_col = _sample_face_color(image, cx, cy, radius)

    diff = cv2.absdiff(image.astype(np.float32), face_col).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    mean_val, std_val = cv2.meanStdDev(diff_gray)
    thresh = max(15, min(50, float(mean_val[0][0]) + 2.0 * float(std_val[0][0])))
    _, binary = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)

    # Morphological noise removal
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # Connected component filtering
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        x, y, wb, hb, area = stats[i]
        long_dim = max(wb, hb)
        short_dim = min(wb, hb)
        is_elongated = long_dim > radius * 0.25
        near_centre = (x - 15 <= cx <= x + wb + 15) and (y - 15 <= cy <= y + hb + 15)
        aspect_ok = short_dim > 0 and (long_dim / short_dim) > 2.5
        if is_elongated and (near_centre or (aspect_ok and area > 80)):
            hand_mask[labels == i] = 255

    # Always protect a pivot circle at centre
    cv2.circle(hand_mask, (cx, cy), max(8, int(radius * 0.10)), 255, -1)

    # Dilate to cover anti-aliased edges
    if dilation > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        hand_mask = cv2.dilate(hand_mask, k_dil, iterations=1)

    # Clip to clock face circle
    face_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(face_circle, (cx, cy), int(radius * 0.88), 255, -1)
    hand_mask = cv2.bitwise_and(hand_mask, face_circle)

    return hand_mask


def detect_existing_hands(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    canny_low: int = 50,
    canny_high: int = 150,
    angle_eps_deg: float = 7.0,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Detect existing clock hands as line candidates using HoughLinesP.

    Strategy
    --------
    1. Crop to a square region around the dial (≈ 1.2 × radius).
    2. Convert to grayscale, lightly blur, run Canny edge detection.
    3. Run cv2.HoughLinesP to get candidate line segments.
    4. Keep only segments that:
       - Have one endpoint very close to the clock centre; and
       - Extend well into the dial but still inside the radius.
    5. Group nearly parallel, nearby segments into one hand candidate each.

    Parameters
    ----------
    image      : RGB ndarray
    cx, cy     : clock centre (pixels)
    radius     : clock radius (pixels)
    canny_low  : low threshold for Canny
    canny_high : high threshold for Canny
    angle_eps_deg : angular tolerance (degrees) when grouping segments

    Returns
    -------
    (hand_candidates, debug_image)

    hand_candidates : list of dicts with keys:
        - 'angle_deg' : float, 0 = 12 o'clock, clockwise
        - 'length'    : float, radial length from centre to tip (pixels)
        - 'p_inner'   : (x, y) pivot-endpoint (near centre)
        - 'p_outer'   : (x, y) tip-endpoint (far from centre)
    debug_image : RGB image with accepted candidate lines drawn on top
    """
    h, w = image.shape[:2]

    # ROI around the dial to avoid spurious lines at the border/background
    roi_r = int(radius * 1.2)
    x0 = max(0, cx - roi_r)
    y0 = max(0, cy - roi_r)
    x1 = min(w, cx + roi_r)
    y1 = min(h, cy + roi_r)

    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return [], image.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Hough parameters scale with radius for robustness across resolutions
    min_line_len = max(10, int(radius * 0.30))
    max_line_gap = max(5, int(radius * 0.08))
    threshold = max(20, int(radius * 0.25))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    hand_candidates: List[Dict[str, float]] = []
    debug = image.copy()

    face_col = _sample_face_color(image, cx, cy, radius)

    def _candidate_mask_and_support(ox: int, oy: int) -> Tuple[np.ndarray, int]:
        """
        Build a refined binary mask for a single candidate using background-diff
        inside a narrow corridor. Returns (mask, support_area_px).
        """
        corridor_w = max(8, int(radius * 0.06))
        corridor = _corridor_mask(image.shape[:2], cx, cy, ox, oy, thickness=corridor_w)

        diff = cv2.absdiff(image.astype(np.float32), face_col).astype(np.uint8)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        vals = diff_gray[corridor > 0]
        if vals.size < 40:
            return np.zeros((h, w), dtype=np.uint8), 0

        # Otsu on corridor pixels, clamped to a sane minimum
        vimg = vals.reshape(-1, 1).astype(np.uint8)
        t, _ = cv2.threshold(vimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(max(12.0, min(80.0, t)))

        binary = np.zeros((h, w), dtype=np.uint8)
        binary[(diff_gray > t) & (corridor > 0)] = 255

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

        # Keep only components that touch the pivot region and extend outward,
        # which helps reject numerals/ticks near the edge.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros((h, w), dtype=np.uint8)
        centre_r = max(6, int(radius * 0.10))
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 60:
                continue
            ys, xs = np.where(labels == i)
            if xs.size == 0:
                continue
            d = np.hypot(xs - cx, ys - cy)
            if (d < centre_r).any() and (d > radius * 0.25).any():
                keep[labels == i] = 255

        if keep.max() == 0:
            return keep, 0

        # Tighten: keep only pixels within the dial
        dial = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dial, (cx, cy), int(radius * 0.92), 255, -1)
        keep = cv2.bitwise_and(keep, dial)
        support = int((keep > 0).sum())
        return keep, support

    def _add_candidate(px1, py1, px2, py2):
        # Ensure (p_inner) is closer to centre, (p_outer) is farther away.
        d1 = math.hypot(px1 - cx, py1 - cy)
        d2 = math.hypot(px2 - cx, py2 - cy)
        if d1 <= d2:
            ix, iy, ox, oy = px1, py1, px2, py2
            inner_r, outer_r = d1, d2
        else:
            ix, iy, ox, oy = px2, py2, px1, py1
            inner_r, outer_r = d2, d1

        # Require inner point to be close to centre and outer point inside dial
        if inner_r > radius * 0.20:
            return
        if outer_r < radius * 0.28 or outer_r > radius * 1.05:
            return

        # Angle convention: 0 = 12 o'clock, clockwise
        dx = ox - cx
        dy = oy - cy
        angle_deg = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0

        cand: Dict[str, float] = {
            "angle_deg": angle_deg,
            "length": float(outer_r),
            "p_inner_x": float(ix),
            "p_inner_y": float(iy),
            "p_outer_x": float(ox),
            "p_outer_y": float(oy),
        }

        # Reject weak candidates that don't have enough hand-like support.
        mask_c, support = _candidate_mask_and_support(int(round(ox)), int(round(oy)))
        if support < max(80, int(radius * 2.0)):
            return
        cand["support_px"] = float(support)
        hand_candidates.append(cand)

    if lines is not None:
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            # Map back to full-image coordinates
            px1 = x1_l + x0
            py1 = y1_l + y0
            px2 = x2_l + x0
            py2 = y2_l + y0
            _add_candidate(px1, py1, px2, py2)

    # If Hough failed to provide candidates, fall back to legacy detector.
    if not hand_candidates:
        legacy_mask = _legacy_detect_existing_hands(image, cx, cy, radius, dilation=0)
        if legacy_mask.max() == 0:
            return [], debug

        num, labels, stats, _ = cv2.connectedComponentsWithStats(legacy_mask, connectivity=8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 40:
                continue
            ys, xs = np.where(labels == i)
            if xs.size == 0:
                continue
            # Fartherst point from centre → approximate tip
            dists = np.hypot(xs - cx, ys - cy)
            idx = int(dists.argmax())
            ox, oy = int(xs[idx]), int(ys[idx])
            _add_candidate(cx, cy, ox, oy)

    if not hand_candidates:
        return [], debug

    # ── Group nearly-parallel, overlapping segments into single hands ───────
    angle_eps = float(angle_eps_deg)
    used = [False] * len(hand_candidates)
    grouped: List[Dict[str, float]] = []

    # Process longer segments first (more reliable orientation)
    order = sorted(range(len(hand_candidates)), key=lambda i: -hand_candidates[i]["length"])

    for idx in order:
        if used[idx]:
            continue
        base = hand_candidates[idx]
        base_angle = base["angle_deg"]
        base_len = base["length"]

        cluster_indices = [idx]
        used[idx] = True

        for j in order:
            if used[j]:
                continue
            cand = hand_candidates[j]
            da = abs((cand["angle_deg"] - base_angle + 180.0) % 360.0 - 180.0)
            if da <= angle_eps:
                # Radial ranges should overlap
                if cand["length"] > radius * 0.25:
                    cluster_indices.append(j)
                    used[j] = True

        # Represent cluster by the longest member
        best_idx = max(cluster_indices, key=lambda k: hand_candidates[k]["length"])
        grouped.append(hand_candidates[best_idx])

    # Draw debug lines
    colours = [
        (0, 220, 0),
        (220, 0, 0),
        (0, 0, 220),
        (220, 180, 0),
        (0, 220, 220),
    ]
    dbg_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
    for i, cand in enumerate(grouped):
        color = colours[i % len(colours)]
        ix = int(round(cand["p_inner_x"]))
        iy = int(round(cand["p_inner_y"]))
        ox = int(round(cand["p_outer_x"]))
        oy = int(round(cand["p_outer_y"]))
        cv2.line(dbg_bgr, (ix, iy), (ox, oy), color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(dbg_bgr, (ix, iy), 3, (255, 255, 255), -1)

    debug_rgb = cv2.cvtColor(dbg_bgr, cv2.COLOR_BGR2RGB)
    return grouped, debug_rgb


def build_hand_mask(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    line_candidates: List[Dict[str, float]],
    dilation: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterise hand line candidates into a binary mask.

    Parameters
    ----------
    image          : RGB ndarray (used only for size)
    cx, cy, radius : clock geometry
    line_candidates: output from detect_existing_hands()
    dilation       : radius of final dilation kernel

    Returns
    -------
    (raw_mask, dilated_mask) - both uint8, 255 = hand pixel
    """
    h, w = image.shape[:2]
    raw = np.zeros((h, w), dtype=np.uint8)

    if not line_candidates:
        return raw, raw.copy()

    face_col = _sample_face_color(image, cx, cy, radius)
    diff = cv2.absdiff(image.astype(np.float32), face_col).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    dial = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(dial, (cx, cy), int(radius * 0.92), 255, -1)

    # Refined mask: threshold within a narrow corridor for each candidate,
    # keep only components that touch the pivot and extend outward.
    centre_r = max(6, int(radius * 0.10))
    corridor_w = max(8, int(radius * 0.06))
    for cand in line_candidates:
        ox = int(round(cand["p_outer_x"]))
        oy = int(round(cand["p_outer_y"]))
        corridor = _corridor_mask((h, w), cx, cy, ox, oy, thickness=corridor_w)
        vals = diff_gray[corridor > 0]
        if vals.size < 40:
            continue

        vimg = vals.reshape(-1, 1).astype(np.uint8)
        t, _ = cv2.threshold(vimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(max(12.0, min(80.0, t)))

        binary = np.zeros((h, w), dtype=np.uint8)
        binary[(diff_gray > t) & (corridor > 0)] = 255

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 60:
                continue
            ys, xs = np.where(labels == i)
            if xs.size == 0:
                continue
            d = np.hypot(xs - cx, ys - cy)
            if (d < centre_r).any() and (d > radius * 0.25).any():
                raw[labels == i] = 255

    raw = cv2.bitwise_and(raw, dial)

    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        dil = cv2.dilate(raw, k, iterations=1)
        dil = cv2.bitwise_and(dil, dial)
    else:
        dil = raw.copy()

    # Binarise anti-aliased edges
    raw[raw > 0] = 255
    dil[dil > 0] = 255

    return raw, dil


# ===========================================================================
# 4.  HAND REMOVAL  (smudge-free: fill → edge-blend)
# ===========================================================================

def remove_hands(
    image: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> np.ndarray:
    """
    Erase clock hands from *image* using a smudge-free two-step approach.

    Step 1  - Direct fill
        Fill all masked pixels with the sampled face background colour.
        This guarantees no dark-hand pixel bleeds into the reconstruction.

    Step 2  - Edge smoothing (TELEA, tiny radius)
        Run TELEA inpainting ONLY on the 4-pixel boundary of the mask.
        TELEA interpolates from the surrounding (now face-coloured) pixels
        → smooth transition, zero smudge.

    Why this beats pure NS inpainting
    -----------------------------------
    NS inpainting propagates from the mask boundary inward.  When the
    boundary still contains dark hand-edge pixels (anti-aliasing bleed),
    the propagation carries those dark values inward, creating the gray
    smudge seen in the previous attempt.  By pre-filling with face colour
    we remove that source of contamination entirely.

    Parameters
    ----------
    image  : RGB ndarray (original, with hands)
    mask   : uint8 binary hand mask (255 = hand pixel)
    cx, cy : clock centre
    radius : clock radius

    Returns
    -------
    RGB ndarray - clean clock face, hands removed
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    # ── Step 1: direct fill ────────────────────────────────────────────────
    face_col = _sample_face_color(image, cx, cy, radius)   # shape (3,)
    result[mask > 0] = face_col

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Step 2: boundary smooth ────────────────────────────────────────────
    # Build a thin boundary ring around the mask
    k_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded   = cv2.erode(mask, k_erode, iterations=1)
    boundary = cv2.bitwise_and(mask, cv2.bitwise_not(eroded))

    # Small 3-px dilation so we blend slightly into the face side too
    k_dil    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(boundary, k_dil, iterations=1)

    if boundary.max() > 0:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(result_bgr, boundary, inpaintRadius=3,
                                  flags=cv2.INPAINT_TELEA)
        result     = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return result


def remove_old_hands(
    image: np.ndarray,
    hand_mask: Optional[np.ndarray],
    cx: int,
    cy: int,
    radius: int,
    extra_dilation: int = 2,
) -> np.ndarray:
    """
    Convenience wrapper for removing existing hands using a pre-built mask.

    Performs a small additional dilation of the incoming *hand_mask* to ensure
    full coverage of anti-aliased edges, then calls :func:`remove_hands`.

    Parameters
    ----------
    image         : RGB ndarray (original watch image)
    hand_mask     : uint8 binary hand mask (255 = hand pixel)
    cx, cy        : clock centre
    radius        : clock radius
    extra_dilation: additional dilation kernel size (px) before inpainting
    """
    if hand_mask is None or hand_mask.size == 0 or hand_mask.max() == 0:
        # Nothing to remove - return a copy of the original.
        return image.copy()

    mask = hand_mask.copy()
    if extra_dilation > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (extra_dilation, extra_dilation)
        )
        mask = cv2.dilate(mask, k, iterations=1)

    return remove_hands(image, mask, cx, cy, radius)


# ===========================================================================
# 5.  HAND DRAWING
# ===========================================================================

def _make_tapered_polygon(
    cx: float, cy: float,
    angle_deg: float,
    tip_len: float,
    tail_len: float,
    base_hw: float,
    tip_hw: float = 0.8,
) -> np.ndarray:
    """
    Build a 6-point tapered clock-hand polygon (int32, shape (6, 2)).

    Shape: wide near the pivot, narrows to a near-point at the tip,
           with a short counterweight tail behind the pivot.
    """
    ar   = math.radians(angle_deg - 90.0)
    fdx  = math.cos(ar)          # forward direction (toward tip)
    fdy  = math.sin(ar)
    pdx  = -fdy                  # perpendicular (left)
    pdy  =  fdx

    # Widest point sits 12 % of tip_len ahead of pivot
    WIDE_FRAC = 0.12
    wx = cx + WIDE_FRAC * tip_len * fdx
    wy = cy + WIDE_FRAC * tip_len * fdy
    tail_hw = max(tip_hw * 0.5, base_hw * 0.25)

    pts = np.array([
        [cx - tail_len * fdx + tail_hw * pdx,   cy - tail_len * fdy + tail_hw * pdy],
        [cx - tail_len * fdx - tail_hw * pdx,   cy - tail_len * fdy - tail_hw * pdy],
        [wx - base_hw * pdx,                     wy - base_hw * pdy],
        [cx + tip_len * fdx - tip_hw * pdx,      cy + tip_len * fdy - tip_hw * pdy],
        [cx + tip_len * fdx + tip_hw * pdx,      cy + tip_len * fdy + tip_hw * pdy],
        [wx + base_hw * pdx,                     wy + base_hw * pdy],
    ], dtype=np.float32)

    return pts.astype(np.int32)


def draw_clock_hand(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
    length: float,
    width: float,
    color: Tuple[int, int, int],
    tail_length: float = 0.0,
    shadow: bool = True,
    shadow_opacity: float = 0.25,
) -> None:
    """
    Draw one tapered clock hand onto *canvas* in-place (BGR uint8).

    Uses a filled polygon (not a line) for a premium look.
    Optionally draws a semi-transparent drop shadow first.

    Parameters
    ----------
    canvas      : BGR ndarray, drawn on in-place
    cx, cy      : clock centre (pixels)
    angle_deg   : hand angle, 0 = 12 o'clock, clockwise
    length      : tip length in pixels
    width       : half-width at widest point (pixels)
    color       : (R, G, B) hand colour
    tail_length : counterweight tail length behind pivot (pixels)
    shadow      : draw drop shadow
    shadow_opacity : alpha of shadow overlay
    """
    tip_hw = max(1.0, width * 0.06)
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    pts = _make_tapered_polygon(cx, cy, angle_deg, length, tail_length, width, tip_hw)

    if shadow:
        shift = max(2, int(width * 0.4))
        shadow_pts = (pts + np.array([shift, shift])).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [shadow_pts], (20, 20, 20))
        cv2.addWeighted(overlay, shadow_opacity, canvas, 1.0 - shadow_opacity, 0, canvas)

    cv2.fillPoly(canvas, [pts], bgr)
    cv2.polylines(canvas, [pts], isClosed=True, color=bgr, thickness=max(1, int(width * 0.15)))


def _estimate_hand_colors_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Estimate (main_hand_color, second_hand_color) in RGB from a hand mask.

    Main colour   → darkest non-vibrant pixels inside the mask
    Second colour → most saturated / reddish pixels inside the mask
    """
    if mask is None or mask.size == 0 or mask.max() == 0:
        return (20, 20, 20), (210, 15, 15)

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tight_mask = cv2.erode(mask, k_erode, iterations=1)
    if tight_mask.max() == 0:
        tight_mask = mask

    sampled_rgb = image[tight_mask > 0]
    if sampled_rgb.size == 0:
        return (20, 20, 20), (210, 15, 15)

    sampled_hsv = cv2.cvtColor(
        sampled_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)

    # Accept only pixels that are actually reddish (hue 0–15 or 160–179 in
    # OpenCV's 0-179 scale) with decent saturation.  This prevents the orange
    # clock face or tan backgrounds from being mistaken for the second hand.
    h_chan  = sampled_hsv[:, 0]   # hue   (0-179)
    sat     = sampled_hsv[:, 1]   # saturation
    val     = sampled_hsv[:, 2]   # value
    reddish = ((h_chan <= 15) | (h_chan >= 160)) & (sat > 80) & (val > 60)
    if reddish.any():
        second_color = tuple(int(c) for c in np.mean(sampled_rgb[reddish], axis=0))
    else:
        second_color = (210, 15, 15)

    gray_vals = cv2.cvtColor(
        sampled_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY
    ).reshape(-1)
    dark = (gray_vals < 120) & (~reddish)
    if dark.any():
        main_color = tuple(int(c) for c in np.median(sampled_rgb[dark], axis=0))
    else:
        main_color = (20, 20, 20)

    return main_color, second_color


def _sample_hand_style(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Backwards-compatible colour sampler used by older callers.

    Uses the new Hough-based detector and mask builder internally, then
    delegates colour estimation to _estimate_hand_colors_from_mask.
    """
    candidates, _ = detect_existing_hands(image, cx, cy, radius)
    raw_mask, _ = build_hand_mask(image, cx, cy, radius, candidates, dilation=0)
    return _estimate_hand_colors_from_mask(image, raw_mask)


def estimate_hand_style(
    image: np.ndarray,
    hand_mask: Optional[np.ndarray],
    cx: int,
    cy: int,
    radius: int,
) -> Dict[str, object]:
    """
    Public helper to estimate hand style (colours, widths, default lengths).

    Parameters
    ----------
    image     : RGB source image
    hand_mask : optional binary mask of existing hands (uint8, 255 = hand)
    cx, cy    : clock centre
    radius    : clock radius

    Returns
    -------
    style dict with keys:
        - 'hour_color', 'minute_color', 'second_color'  (RGB tuples)
        - 'hour_width', 'minute_width', 'second_width'  (half-width, px)
        - 'hour_length_ratio', 'minute_length_ratio', 'second_length_ratio'
    """
    if hand_mask is None or hand_mask.max() == 0:
        main_color, second_color = _sample_hand_style(image, cx, cy, radius)
        mask_for_thickness = None
    else:
        main_color, second_color = _estimate_hand_colors_from_mask(image, hand_mask)
        mask_for_thickness = hand_mask

    # ── Contrast guard ────────────────────────────────────────────────────
    # If the sampled hand color is too similar to the face color the hands
    # will be invisible after redrawing.  Fall back to a contrasting color.
    face_col_f = np.array(_sample_face_color(image, cx, cy, radius), dtype=float)
    main_col_f = np.array(main_color, dtype=float)
    face_lum   = float(0.299*face_col_f[0] + 0.587*face_col_f[1] + 0.114*face_col_f[2])
    main_lum   = float(0.299*main_col_f[0] + 0.587*main_col_f[1] + 0.114*main_col_f[2])

    if abs(main_lum - face_lum) < 45:
        if face_lum < 128:
            # Dark face → sample the bright outer ring (border) as hand color
            h_img, w_img = image.shape[:2]
            ring_pixels = []
            for deg in range(0, 360, 5):
                rad = math.radians(deg - 90)
                for r_frac in (0.88, 0.93, 0.97):
                    rx = int(cx + radius * r_frac * math.cos(rad))
                    ry = int(cy + radius * r_frac * math.sin(rad))
                    if 0 <= rx < w_img and 0 <= ry < h_img:
                        ring_pixels.append(image[ry, rx])
            if ring_pixels:
                ring_arr = np.array(ring_pixels, dtype=float)
                # Take the brightest quartile - that's the border colour
                brightness = ring_arr.mean(axis=1)
                bright_idx = brightness >= np.percentile(brightness, 75)
                main_color = tuple(int(v) for v in ring_arr[bright_idx].mean(axis=0))
            else:
                main_color = (220, 200, 160)
        else:
            # Light face → dark hands
            main_color = (30, 30, 30)

    # Thickness estimation from connected components when a mask is available
    est_thickness = 4
    if mask_for_thickness is not None and mask_for_thickness.max() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_for_thickness, connectivity=8
        )
        thicknesses: List[int] = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 40:
                continue
            wb = stats[i, cv2.CC_STAT_WIDTH]
            hb = stats[i, cv2.CC_STAT_HEIGHT]
            thicknesses.append(int(min(wb, hb)))

        if thicknesses:
            est_thickness = int(np.median(thicknesses))
            est_thickness = max(3, min(8, est_thickness))

    hour_width = max(est_thickness + 1, int(radius * 0.055))
    minute_width = max(est_thickness, int(radius * 0.038))
    second_width = max(1, int(radius * 0.016))

    style: Dict[str, object] = {
        "hour_color": main_color,
        "minute_color": main_color,
        "second_color": second_color,
        "hour_width": hour_width,
        "minute_width": minute_width,
        "second_width": second_width,
        # Length ratios follow the specification defaults
        "hour_length_ratio": 0.50,
        "minute_length_ratio": 0.75,
        "second_length_ratio": 0.85,
    }
    return style


# ===========================================================================
# 6.  RENDER NEW HANDS  (SSAA)
# ===========================================================================

# Hand geometry as fractions of clock radius
_HAND_PARAMS = {
    #            tip_ratio  tail_ratio  hw_ratio
    'hour':     (0.50,      0.10,       0.055),
    'minute':   (0.75,      0.13,       0.038),
    'second':   (0.85,      0.22,       0.016),
}


def render_new_hands(
    face_image: np.ndarray,
    hour_angle: float,
    minute_angle: float,
    second_angle: float,
    cx: int,
    cy: int,
    radius: int,
    hour_color: Tuple[int, int, int] = (20, 20, 20),
    minute_color: Tuple[int, int, int] = (20, 20, 20),
    second_color: Tuple[int, int, int] = (210, 15, 15),
    hour_width: Optional[int] = None,
    minute_width: Optional[int] = None,
    second_width: Optional[int] = None,
) -> np.ndarray:
    """
    Draw hour, minute, and second hands onto *face_image* using 4× SSAA.

    All size parameters default to proportional-to-radius values when None.
    Draw order: hour → minute → second → hub cap  (correct depth order).

    Parameters
    ----------
    face_image       : RGB ndarray - clean clock face (hands already removed)
    *_angle          : hand angles from compute_hand_angles()
    cx, cy, radius   : from detect_watch_circle()
    *_color          : (R, G, B) for each hand
    *_width          : half-width in pixels (None → auto from radius)

    Returns
    -------
    RGB ndarray - final clock image with new hands
    """
    h, w = face_image.shape[:2]
    SF   = 4                   # super-sampling factor
    bh, bw = h * SF, w * SF
    bcx, bcy, br = cx * SF, cy * SF, radius * SF

    # Default widths proportional to radius
    h_hw = int(hour_width   or max(4, int(radius * _HAND_PARAMS['hour'][2])))
    m_hw = int(minute_width or max(3, int(radius * _HAND_PARAMS['minute'][2])))
    s_hw = int(second_width or max(2, int(radius * _HAND_PARAMS['second'][2])))

    # Upscale canvas (BGR)
    canvas = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    big    = cv2.resize(canvas, (bw, bh), interpolation=cv2.INTER_CUBIC)

    def _draw(angle, key, hw, color):
        tip_r, tail_r, _ = _HAND_PARAMS[key]
        tip  = br * tip_r
        tail = br * tail_r
        draw_clock_hand(big, bcx, bcy, angle,
                        length=tip, width=hw * SF,
                        color=color, tail_length=tail,
                        shadow=True, shadow_opacity=0.22)

    _draw(hour_angle,   'hour',   h_hw, hour_color)
    _draw(minute_angle, 'minute', m_hw, minute_color)
    _draw(second_angle, 'second', s_hw, second_color)

    # Hub cap - sits on top of all hands
    hub_r = max(SF * 3, int(br * 0.038))
    h_bgr = (int(hour_color[2]), int(hour_color[1]), int(hour_color[0]))
    dark  = tuple(max(0, c - 50) for c in h_bgr)
    cv2.circle(big, (bcx, bcy), hub_r + SF + 1, dark,  -1)
    cv2.circle(big, (bcx, bcy), hub_r,           h_bgr, -1)
    gx, gy, gr = bcx - hub_r // 3, bcy - hub_r // 3, max(1, hub_r // 3)
    cv2.circle(big, (gx, gy), gr, (220, 220, 220), -1)

    # Downsample with area averaging (best for downscale AA)
    result = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def draw_new_hands(
    clean_face: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    target_time: Tuple[int, int, int],
    style: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """
    High-level helper: draw new hands for *target_time* onto a clean face.

    This wraps :func:`compute_hand_angles` and :func:`render_new_hands`,
    taking a style dictionary as produced by :func:`estimate_hand_style`.

    Parameters
    ----------
    clean_face : RGB ndarray with old hands already removed
    cx, cy     : clock centre
    radius     : clock radius
    target_time: (hour, minute, second)
    style      : optional dict from estimate_hand_style(); if None a fresh
                 style will be estimated from *clean_face* itself.
    """
    h, m, s = parse_time(*target_time)
    hour_ang, minute_ang, second_ang = compute_hand_angles(h, m, s)

    if style is None:
        style = estimate_hand_style(clean_face, None, cx, cy, radius)

    hour_color = tuple(style.get("hour_color", (20, 20, 20)))  # type: ignore[arg-type]
    minute_color = tuple(style.get("minute_color", (20, 20, 20)))  # type: ignore[arg-type]
    second_color = tuple(style.get("second_color", (210, 15, 15)))  # type: ignore[arg-type]

    hour_width = int(style.get("hour_width", max(4, int(radius * _HAND_PARAMS["hour"][2]))))  # type: ignore[arg-type]
    minute_width = int(style.get("minute_width", max(3, int(radius * _HAND_PARAMS["minute"][2]))))  # type: ignore[arg-type]
    second_width = int(style.get("second_width", max(2, int(radius * _HAND_PARAMS["second"][2]))))  # type: ignore[arg-type]

    return render_new_hands(
        clean_face,
        hour_ang,
        minute_ang,
        second_ang,
        cx,
        cy,
        radius,
        hour_color=hour_color,
        minute_color=minute_color,
        second_color=second_color,
        hour_width=hour_width,
        minute_width=minute_width,
        second_width=second_width,
    )


def _wrap_angle_deg(angle: float) -> float:
    return float(angle) % 360.0


def _signed_angle_delta_deg(target: float, current: float) -> float:
    """
    Smallest signed delta from current → target in degrees (clockwise-positive
    in our convention, but this is purely numeric).
    """
    t = _wrap_angle_deg(target)
    c = _wrap_angle_deg(current)
    return ((t - c + 180.0) % 360.0) - 180.0


def _classify_hand_candidates(
    candidates: List[Dict[str, float]],
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Assign detected candidates to hour/minute/second by length (radial extent).

    Assumption: second > minute > hour in length when all exist. If only two
    hands are detected, treat them as minute (longer) and hour (shorter).
    """
    if not candidates:
        return {"hour": None, "minute": None, "second": None}

    # Prefer stronger candidates (support) when available
    ordered = sorted(
        candidates,
        key=lambda c: (float(c.get("support_px", 0.0)), float(c.get("length", 0.0))),
    )
    if len(ordered) == 1:
        return {"hour": ordered[0], "minute": None, "second": None}
    if len(ordered) == 2:
        return {"hour": ordered[0], "minute": ordered[1], "second": None}

    # 3+ candidates: take the top 3 by length
    top3 = ordered[-3:]
    hour = top3[0]
    minute = top3[1]
    second = top3[2]
    return {"hour": hour, "minute": minute, "second": second}


def _red_score_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Heuristic: return how "second-hand-red" the masked pixels are.
    Value in [0, 1]-ish. Higher means more red-like pixels.
    """
    if mask is None or mask.max() == 0:
        return 0.0
    pixels = image_rgb[mask > 0]
    if pixels.size == 0:
        return 0.0
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.int32)
    s = hsv[:, 1].astype(np.int32)
    v = hsv[:, 2].astype(np.int32)
    red_hue = (h <= 12) | (h >= 168)
    vivid = (s >= 70) & (v >= 60)
    redish = red_hue & vivid
    frac = float(np.mean(redish)) if redish.size else 0.0
    sat = float(np.mean(s[vivid])) / 255.0 if vivid.any() else 0.0
    return frac * 0.8 + sat * 0.2


def _refined_candidate_mask(
    image_rgb: np.ndarray,
    face_color: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    cand: Dict[str, float],
    corridor_w: Optional[int] = None,
) -> np.ndarray:
    """
    Build a refined binary mask for a specific candidate using background-diff
    inside a narrow corridor + connected-component constraints.
    """
    h, w = image_rgb.shape[:2]
    ox = int(round(float(cand["p_outer_x"])))
    oy = int(round(float(cand["p_outer_y"])))
    cw = int(corridor_w if corridor_w is not None else max(8, int(radius * 0.06)))
    corridor = _corridor_mask((h, w), cx, cy, ox, oy, thickness=cw)

    diff = cv2.absdiff(image_rgb.astype(np.float32), face_color).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    vals = diff_gray[corridor > 0]
    if vals.size < 40:
        return np.zeros((h, w), dtype=np.uint8)

    vimg = vals.reshape(-1, 1).astype(np.uint8)
    t, _ = cv2.threshold(vimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(max(12.0, min(80.0, t)))

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[(diff_gray > t) & (corridor > 0)] = 255

    # Prefer not to bridge far-away ticks/numerals: use a smaller close kernel.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros((h, w), dtype=np.uint8)
    centre_r = max(6, int(radius * 0.10))
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        d = np.hypot(xs - cx, ys - cy)
        if (d < centre_r).any() and (d > radius * 0.25).any():
            keep[labels == i] = 255

    dial = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(dial, (cx, cy), int(radius * 0.92), 255, -1)
    keep = cv2.bitwise_and(keep, dial)
    keep[keep > 0] = 255
    return keep


def _assign_hands_best_effort(
    candidates: List[Dict[str, float]],
    image_rgb: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Assign candidates to hour/minute/second.

    - Second: pick the most red-like candidate (if clearly red).
    - Remaining: shorter = hour, longer = minute.
    """
    if not candidates:
        return {"hour": None, "minute": None, "second": None}

    face_col = _sample_face_color(image_rgb, cx, cy, radius)
    scored = []
    for c in candidates:
        m = _refined_candidate_mask(image_rgb, face_col, cx, cy, radius, c)
        score = _red_score_from_mask(image_rgb, m)
        scored.append((score, float(c.get("length", 0.0)), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    second = None
    if scored and scored[0][0] >= 0.18:
        second = scored[0][2]
        remaining = [t[2] for t in scored[1:]]
    else:
        remaining = [t[2] for t in scored]

    if len(remaining) == 0:
        return {"hour": None, "minute": None, "second": second}
    if len(remaining) == 1:
        return {"hour": remaining[0], "minute": None, "second": second}

    remaining_sorted = sorted(remaining, key=lambda c: float(c.get("length", 0.0)))
    hour = remaining_sorted[0]
    minute = remaining_sorted[-1]
    return {"hour": hour, "minute": minute, "second": second}


def _corridor_mask(
    shape_hw: Tuple[int, int],
    cx: int,
    cy: int,
    ox: int,
    oy: int,
    thickness: int,
) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.line(m, (cx, cy), (ox, oy), 255, thickness=thickness, lineType=cv2.LINE_AA)
    m[m > 0] = 255
    return m


def _extract_hand_layer_premultiplied(
    image_rgb: np.ndarray,
    face_color: np.ndarray,
    corridor: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a hand RGBA-like layer as (premultiplied_rgb, alpha) from the original.

    The extraction is constrained to a wide corridor around a hand candidate,
    and uses background-difference to isolate hand pixels.

    Returns
    -------
    premul_rgb : float32 HxWx3, premultiplied by alpha in [0,1]
    alpha      : float32 HxW in [0,1]
    """
    h, w = image_rgb.shape[:2]
    diff = cv2.absdiff(image_rgb.astype(np.float32), face_color).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    roi_vals = diff_gray[corridor > 0]
    if roi_vals.size == 0:
        alpha = np.zeros((h, w), dtype=np.float32)
        return np.zeros((h, w, 3), dtype=np.float32), alpha

    mean_val, std_val = cv2.meanStdDev(roi_vals.reshape(-1, 1))
    thresh = max(12.0, min(60.0, float(mean_val[0][0]) + 1.8 * float(std_val[0][0])))

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[(diff_gray > thresh) & (corridor > 0)] = 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # Keep only components that plausibly belong to a hand (touch centre & extend outward)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros((h, w), dtype=np.uint8)
    centre_r = max(6, int(radius * 0.10))
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        d = np.hypot(xs - cx, ys - cy)
        touches_centre = (d < centre_r).any()
        extends_out = (d > radius * 0.25).any()
        if touches_centre and extends_out:
            keep[labels == i] = 255

    if keep.max() == 0:
        keep = binary

    # Feather only slightly at the boundary (avoid expanding into pure background)
    keep_dil = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    dist = cv2.distanceTransform(keep_dil, cv2.DIST_L2, 3).astype(np.float32)
    feather = 2.2
    alpha = np.clip(dist / feather, 0.0, 1.0)
    alpha[keep_dil == 0] = 0.0

    premul_rgb = image_rgb.astype(np.float32) * alpha[..., None]
    return premul_rgb, alpha


def _rotate_layer_about_center(
    premul_rgb: np.ndarray,
    alpha: np.ndarray,
    cx: int,
    cy: int,
    delta_clockwise_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a premultiplied layer (premul_rgb, alpha) about (cx,cy).

    OpenCV rotation angle is counter-clockwise, so we negate clockwise delta.
    """
    angle_ccw = -float(delta_clockwise_deg)
    M = cv2.getRotationMatrix2D((cx, cy), angle_ccw, 1.0)
    h, w = alpha.shape[:2]
    a2 = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb2 = cv2.warpAffine(premul_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb2, a2


def render_hands_by_reusing_original(
    original_rgb: np.ndarray,
    clean_face_rgb: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    candidates: List[Dict[str, float]],
    target_angles: Tuple[float, float, float],
) -> np.ndarray:
    """
    Render new hands by reusing original-hand pixels (extract → rotate → composite).

    This produces the closest match to the original hand style, since the
    pixels come directly from the input image.
    """
    face_color = _sample_face_color(original_rgb, cx, cy, radius)
    hands = _assign_hands_best_effort(candidates, original_rgb, cx, cy, radius)

    out = clean_face_rgb.astype(np.float32)
    hour_t, minute_t, second_t = target_angles

    # Extract a hub cap patch from the original so it looks identical
    hub_r = max(10, int(radius * 0.08))
    hub_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    cv2.circle(hub_mask, (cx, cy), hub_r, 255, -1)
    hub_patch = original_rgb.copy()

    def _apply(kind: str, target_angle: float):
        cand = hands.get(kind)
        if cand is None:
            return
        ox = int(round(cand["p_outer_x"]))
        oy = int(round(cand["p_outer_y"]))
        current_angle = float(cand["angle_deg"])
        delta = _signed_angle_delta_deg(target_angle, current_angle)

        # Use a tighter refined mask to avoid dragging ticks/numerals.
        refined = _refined_candidate_mask(original_rgb, face_color, cx, cy, radius, cand, corridor_w=max(8, int(radius * 0.06)))
        refined = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        corridor = refined  # compatible with _extract_hand_layer_premultiplied signature
        premul, alpha = _extract_hand_layer_premultiplied(original_rgb, face_color, corridor, cx, cy, radius)

        premul_r, alpha_r = _rotate_layer_about_center(premul, alpha, cx, cy, delta_clockwise_deg=delta)
        out[:] = out * (1.0 - alpha_r[..., None]) + premul_r

    # Depth order: hour → minute → second
    _apply("hour", hour_t)
    _apply("minute", minute_t)
    _apply("second", second_t)

    # Restore the original hub cap on top (preserves centre cap style)
    out_u8 = out.clip(0, 255).astype(np.uint8)
    out_u8[hub_mask > 0] = hub_patch[hub_mask > 0]
    return out_u8


# ===========================================================================
# 7.  DEBUG OUTPUT HELPERS
# ===========================================================================

def _save_debug(image_rgb: np.ndarray, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    print(f"  [debug] {label}: {path}")


def _draw_circle_debug(image: np.ndarray, cx, cy, radius) -> np.ndarray:
    out = image.copy()
    cv2.circle(out, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
    return out


def _draw_mask_debug(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the hand mask in red on the original image."""
    out = image.copy()
    out[mask > 0] = (out[mask > 0] * 0.4 + np.array([200, 30, 30]) * 0.6).astype(np.uint8)
    return out


# ===========================================================================
# 8.  MAIN PIPELINE
# ===========================================================================

def process_watch_pair(
    time_tuple: Tuple[int, int, int],
    analog_image: np.ndarray,
    output_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
    source_image_for_style: Optional[np.ndarray] = None,
    reuse_original_hands: bool = False,
    override_angles: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Full end-to-end pipeline: given a time and an analog watch image,
    return a new image with the hands repositioned to show that time.

    Parameters
    ----------
    time_tuple              : (hour, minute, second)
    analog_image            : RGB ndarray of the analog watch
    output_path             : where to save the result (None = don't save)
    debug_dir               : directory for intermediate debug images
                              (None = skip debug output)
    source_image_for_style  : image from which to sample hand colours/widths
                              (defaults to analog_image itself)
    override_angles         : (hour_deg, minute_deg, second_deg) predicted by
                              Stage 2 neural network.  When provided these take
                              priority over the analytic formula so the full
                              pipeline stays network-based (requirement 6).

    Returns
    -------
    RGB ndarray - the edited watch image

    Pipeline Steps
    --------------
    1. parse_time             - validate input
    2. detect_watch_circle    - find (cx, cy, radius)
    3. compute_hand_angles    - get (h_ang, m_ang, s_ang)
    4. detect_existing_hands  - binary mask of old hands
    5. remove_hands           - fill + edge-blend (smudge-free)
    6. estimate_hand_style    - get colours/widths from source image
    7. render_new_hands       - SSAA draw at 4× then downsample
    8. save output + debug images
    """
    # ── 1. Parse time ────────────────────────────────────────────────────
    h, m, s = parse_time(*time_tuple)
    print(f"[hand_editor] Target time: {h:02d}:{m:02d}:{s:02d}")

    # ── 2. Detect circle ─────────────────────────────────────────────────
    cx, cy, radius = detect_watch_circle(analog_image)
    print(f"[hand_editor] Watch circle: centre=({cx},{cy}), radius={radius}px")

    style_src = source_image_for_style if source_image_for_style is not None else analog_image

    dbg: Optional[Path] = None
    if debug_dir:
        dbg = Path(debug_dir)
        _save_debug(
            _draw_circle_debug(analog_image, cx, cy, radius),
            dbg / "detected_circle.png",
            "detected circle",
        )

    # ── 3. Compute angles ────────────────────────────────────────────────
    if override_angles is not None:
        h_ang, m_ang, s_ang = override_angles
        formula_h, formula_m, formula_s = compute_hand_angles(h, m, s)
        print(
            f"[hand_editor] Angles (Stage 2 network): H={h_ang:.1f}° M={m_ang:.1f}° S={s_ang:.1f}°"
            f"  |  formula: H={formula_h:.1f}° M={formula_m:.1f}° S={formula_s:.1f}°"
        )
    else:
        h_ang, m_ang, s_ang = compute_hand_angles(h, m, s)
        print(f"[hand_editor] Angles (formula): H={h_ang:.1f}° M={m_ang:.1f}° S={s_ang:.1f}°")

    # ── 4. Detect old hands (Hough-based) ────────────────────────────────
    hand_candidates, lines_dbg = detect_existing_hands(analog_image, cx, cy, radius)

    if dbg is not None:
        _save_debug(
            lines_dbg,
            dbg / "detected_lines.png",
            "detected hand lines",
        )

    mask_dilation = 3 if reuse_original_hands else 7
    raw_mask, dil_mask = build_hand_mask(
        analog_image, cx, cy, radius, hand_candidates, dilation=mask_dilation
    )

    if dbg is not None:
        _save_debug(
            _draw_mask_debug(analog_image, raw_mask),
            dbg / "hand_mask.png",
            "hand mask (raw)",
        )
        _save_debug(
            _draw_mask_debug(analog_image, dil_mask),
            dbg / "hand_mask_dilated.png",
            "hand mask (dilated)",
        )

    # ── 5. Remove old hands ──────────────────────────────────────────────
    # For reuse_original_hands we keep only a very tight protected disc
    # around the pivot (approx. hub size) so we don't over-smudge the
    # exact centre, but we still fully erase old hand shafts elsewhere.
    inpaint_mask = dil_mask.copy()
    if reuse_original_hands:
        protect_r = max(6, int(radius * 0.07))
        cv2.circle(inpaint_mask, (cx, cy), protect_r, 0, -1)

    face_clean = remove_old_hands(
        analog_image,
        inpaint_mask,
        cx,
        cy,
        radius,
        extra_dilation=0 if reuse_original_hands else 2,
    )

    if dbg is not None:
        _save_debug(face_clean, dbg / "cleaned_face.png", "clean face")

    # ── 6. Sample hand style ─────────────────────────────────────────────
    style = estimate_hand_style(style_src, raw_mask, cx, cy, radius)
    main_col = tuple(style["hour_color"])  # type: ignore[index]
    sec_col = tuple(style["second_color"])  # type: ignore[index]
    h_hw = int(style["hour_width"])  # type: ignore[index]
    m_hw = int(style["minute_width"])  # type: ignore[index]
    s_hw = int(style["second_width"])  # type: ignore[index]
    print(f"[hand_editor] Hand colours: main={main_col}, second={sec_col}")

    # ── 7. Render new hands ──────────────────────────────────────────────
    if reuse_original_hands:
        result = render_hands_by_reusing_original(
            original_rgb=analog_image,
            clean_face_rgb=face_clean,
            cx=cx,
            cy=cy,
            radius=radius,
            candidates=hand_candidates,
            target_angles=(h_ang, m_ang, s_ang),
        )
    else:
        result = render_new_hands(
            face_clean,
            h_ang,
            m_ang,
            s_ang,
            cx,
            cy,
            radius,
            hour_color=main_col,
            minute_color=main_col,
            second_color=sec_col,
            hour_width=h_hw,
            minute_width=m_hw,
            second_width=s_hw,
        )

    if dbg is not None:
        _save_debug(result, dbg / "final_replaced_hands.png", "final result")

    # ── 8. Save output ───────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), out_bgr)
        print(f"[hand_editor] Saved → {output_path}")

    return result


def process_analog_hand_replacement(
    image_path: str,
    hour: int,
    minute: int,
    second: int,
    output_dir: str,
    output_filename: str = "final_replaced_hands.png",
    reuse_original_hands: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper for the analog-hand-replacement stage only.

    Parameters
    ----------
    image_path      : path to the analog watch image
    hour, minute, second : target time components
    output_dir      : directory where debug outputs and final image are saved
    output_filename : name of the final image file inside *output_dir*

    Saves
    -----
    In *output_dir*:
        - detected_circle.png
        - detected_lines.png
        - hand_mask.png
        - hand_mask_dilated.png
        - cleaned_face.png
        - final_replaced_hands.png  (or *output_filename*)
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Analog image not found: {image_path}")

    analog_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    final_path = out_dir_path / output_filename

    result = process_watch_pair(
        time_tuple=(hour, minute, second),
        analog_image=analog_rgb,
        output_path=str(final_path),
        debug_dir=str(out_dir_path),
        source_image_for_style=analog_rgb,
        reuse_original_hands=reuse_original_hands,
    )

    return result
