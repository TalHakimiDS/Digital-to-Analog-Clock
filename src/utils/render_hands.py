"""
Utility functions for rendering clock hands using OpenCV.

Senior-level rendering pipeline:
  1. Auto-detect true inner clock radius (handles thick-bordered clocks)
  2. Erase existing hands via two-pass adaptive inpainting (no residue)
  3. Sample hand style (color, thickness) from the original clock image
  4. Draw brand-new hands from scratch at 4× resolution (SSAA)
     - Tapered polygon geometry (wide at pivot, sharp tip)
     - Drop shadow for depth
     - Counterweight tail on all hands
     - Hub-cap circle on top of all hands
  5. Downsample back with INTER_AREA for crisp anti-aliased edges
"""

import numpy as np
import cv2
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Low-level helpers (kept from original)
# ---------------------------------------------------------------------------

def _detect_inner_radius(image: np.ndarray, center: Tuple[int, int], rough_radius: int) -> int:
    """
    Use HoughCircles to detect the true inner face radius of the clock,
    ignoring thick decorative borders.  Falls back to rough_radius if
    detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_r = int(rough_radius * 0.55)
    max_r = int(rough_radius * 1.05)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=rough_radius,
        param1=60, param2=30,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return rough_radius
    circles = np.round(circles[0]).astype(int)
    cx, cy = center
    # Pick the circle whose centre is closest to the image centre
    best = min(circles, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    return int(best[2])


def _sample_face_color(image: np.ndarray, center: Tuple[int, int], radius: int) -> Tuple[int, int, int]:
    """
    Sample the dominant clock face colour from a tight inner annulus
    (20 %–40 % of radius) that sits well inside the numeral ring and
    far from the border.  Uses the MODE (most-common colour) of the
    sampled pixels so a few hand pixels cannot bias the result.
    Returns an (R, G, B) tuple.
    """
    h, w = image.shape[:2]
    y0, x0 = center[1], center[0]
    inner = max(1, int(radius * 0.20))
    outer = int(radius * 0.40)
    pixels = []
    for r in range(inner, outer + 1):
        for theta_deg in range(0, 360, 6):
            rad = math.radians(theta_deg - 90)
            x = int(x0 + r * math.cos(rad))
            y = int(y0 + r * math.sin(rad))
            if 0 <= x < w and 0 <= y < h:
                pixels.append(image[y, x])
    if not pixels:
        return (240, 240, 240)
    arr = np.array(pixels)          # shape (N, 3)
    # Quantise to 8-bin buckets, then find the most common bucket
    quantised = (arr // 32).astype(np.int32)
    keys, counts = np.unique(quantised.view(np.dtype((np.void, quantised.dtype.itemsize * 3))), return_counts=True)
    dominant_bucket = np.frombuffer(keys[np.argmax(counts)], dtype=np.int32) * 32 + 16
    # Refine: average of pixels in that bucket
    mask = np.all(quantised == dominant_bucket, axis=1)
    refined = arr[mask].mean(axis=0) if mask.any() else arr.mean(axis=0)
    return tuple(int(v) for v in refined)


def _find_pivot(image: np.ndarray, center_guess: Tuple[int, int], radius: int) -> Tuple[int, int]:
    """
    Refine the clock pivot by finding the intersection of major hand lines
    using HoughLinesP. Falls back to center_guess when lines are ambiguous.
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=radius // 4,
        minLineLength=radius // 3,
        maxLineGap=10,
    )
    if lines is None:
        return center_guess

    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        dist = abs(a * center_guess[0] + b * center_guess[1] + c) / math.sqrt(a * a + b * b + 1e-6)
        if dist < radius * 0.15:
            valid_lines.append((a, b, c))

    if len(valid_lines) < 2:
        return center_guess

    a1, b1, c1 = valid_lines[0]
    a2, b2, c2 = valid_lines[1]
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return center_guess

    nx = (b1 * c2 - b2 * c1) / det
    ny = (a2 * c1 - a1 * c2) / det

    if math.sqrt((nx - center_guess[0]) ** 2 + (ny - center_guess[1]) ** 2) > radius * 0.2:
        return center_guess

    return (int(nx), int(ny))


def get_hand_mask(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    cover_radius_ratio: float = 0.85,
    mask_dilation: int = 3,
    protect_pivot: bool = True,
) -> np.ndarray:
    """
    Detect clock hands using adaptive background subtraction and connected
    components. Filters out numerals and tick marks by geometry.
    """
    h, w = image.shape[:2]
    cx, cy = center

    face_color = _sample_face_color(image, center, radius)
    diff       = cv2.absdiff(image, np.array(face_color, dtype=np.uint8))
    diff_gray  = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    mean_val, std_val = cv2.meanStdDev(diff_gray)
    thresh = max(15, min(50, mean_val[0][0] + 2.0 * std_val[0][0]))
    _, diff_bin = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_bin    = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)
    hand_mask = np.zeros(diff_bin.shape, dtype=np.uint8)

    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        touches_center = (x - 15 <= cx <= x + w_box + 15) and (y - 15 <= cy <= y + h_box + 15)
        is_long        = max(w_box, h_box) > radius * 0.25
        if (touches_center and is_long) or (is_long and area > 100):
            hand_mask[labels == i] = 255

    if protect_pivot:
        cv2.circle(hand_mask, (cx, cy), max(8, int(radius * 0.1)), 255, -1)

    if mask_dilation > 0:
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilation, mask_dilation))
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), int(radius * cover_radius_ratio), 255, -1)
    hand_mask = cv2.bitwise_and(hand_mask, circle_mask)

    return hand_mask


def _sample_hand_style(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    """
    Infer the clock's hand colours (main vs. second) and approx thickness
    from the source image. Uses eroded hand mask to avoid face-colour bleed.
    Returns:  (main_color_rgb, second_color_rgb, thickness_px)
    """
    hand_mask = get_hand_mask(image, center, radius)
    if hand_mask.max() == 0:
        return (30, 30, 30), (210, 10, 10), 4

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tight_mask   = cv2.erode(hand_mask, kernel_erode, iterations=1)
    if tight_mask.max() == 0:
        tight_mask = hand_mask

    mask_pixels  = tight_mask > 0
    sampled_hsv  = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[mask_pixels]
    sampled_rgb  = image[mask_pixels]

    # Separate vibrant (coloured) pixels → second-hand
    vibrant_idx = (sampled_hsv[:, 1] > 60) & (sampled_hsv[:, 2] > 50)
    if np.any(vibrant_idx):
        second_color = tuple(int(c) for c in np.mean(sampled_rgb[vibrant_idx], axis=0))
    else:
        second_color = (210, 10, 10)

    # Main (dark) hand colour
    gray_vals      = cv2.cvtColor(sampled_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
    dark_idx       = (gray_vals < 120) & (~vibrant_idx)
    if np.any(dark_idx):
        main_color = tuple(int(c) for c in np.median(sampled_rgb[dark_idx], axis=0))
    else:
        main_color = (30, 30, 30)

    # Estimate thickness
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hand_mask, connectivity=8)
    thicknesses = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 40:
            continue
        thicknesses.append(min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))

    thickness = int(np.median(thicknesses)) if thicknesses else 4
    thickness = max(3, min(6, thickness))

    return main_color, second_color, thickness


# ---------------------------------------------------------------------------
# Hand removal / inpainting helpers
# ---------------------------------------------------------------------------

def erase_hands_with_hough(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    inpaint_radius: int = 5,
    **kwargs,
) -> np.ndarray:
    """
    Two-pass Navier-Stokes inpainting for clean hand removal.

    Pass 1 (coarse): use a generously dilated mask to catch thick/dark hands
                     fully.  A large inpaint radius fills the region with
                     approximate background texture.
    Pass 2 (fine):   re-detect residual discolouration and do a tight second
                     inpaint to remove any grey smudge left over from pass 1.

    This two-pass strategy reliably handles both thin marker hands (synthetic
    clocks) and thick, dark hands on real analogue watches.
    """
    img = image.copy()
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2 - 20

    # ── Pass 1: coarse removal with generous dilation ─────────────────────
    hand_mask = get_hand_mask(img, center, radius, mask_dilation=7, **kwargs)
    if hand_mask.max() == 0:
        return img

    img_bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result   = cv2.inpaint(img_bgr, hand_mask, inpaintRadius=8, flags=cv2.INPAINT_NS)

    # ── Pass 2: clean up residual smudge ──────────────────────────────────
    # Search in a generously expanded region around the original hand mask
    # (residue can bleed slightly outside the original mask boundary).
    kernel_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    search_zone = cv2.dilate(hand_mask, kernel_wide, iterations=2)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    face_col   = _sample_face_color(result_rgb, center, radius)
    diff       = cv2.absdiff(result_rgb, np.array(face_col, dtype=np.uint8))
    diff_gray  = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    residue_mask = np.zeros((h, w), dtype=np.uint8)
    residue_mask[diff_gray > 14] = 255         # lower threshold → catches faint smudges
    residue_mask = cv2.bitwise_and(residue_mask, search_zone)

    # Morphological closing: bridge isolated residue pixels into solid blobs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    residue_mask = cv2.morphologyEx(residue_mask, cv2.MORPH_CLOSE, kernel_close)
    residue_mask = cv2.dilate(residue_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

    if residue_mask.max() > 0:
        result = cv2.inpaint(result, residue_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)

    # ── Pass 3: check the center zone specifically (hub area smudge) ───────
    # The crossing point of two hands is the hardest area for inpainting.
    hub_r = max(12, int(radius * 0.15))
    hub_zone = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hub_zone, center, hub_r, 255, -1)
    result_rgb2  = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    face_col2    = _sample_face_color(result_rgb2, center, radius)
    diff2        = cv2.absdiff(result_rgb2, np.array(face_col2, dtype=np.uint8))
    diff_gray2   = cv2.cvtColor(diff2, cv2.COLOR_RGB2GRAY)
    hub_residue  = np.zeros((h, w), dtype=np.uint8)
    hub_residue[diff_gray2 > 12] = 255
    hub_residue  = cv2.bitwise_and(hub_residue, hub_zone)
    if hub_residue.max() > 0:
        result = cv2.inpaint(result, hub_residue, inpaintRadius=5, flags=cv2.INPAINT_NS)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def erase_hands_with_face_color(image, center=None, radius=None, **kwargs):
    """Backward-compatible alias for erase_hands_with_hough."""
    return erase_hands_with_hough(image, center=center, radius=radius)


# ---------------------------------------------------------------------------
# NEW: Senior-Level Draw-From-Scratch Rendering
# ---------------------------------------------------------------------------

def _make_hand_polygon(
    cx: float,
    cy: float,
    angle_deg: float,
    tip_len: float,
    tail_len: float,
    base_hw: float,
    tip_hw: float = 0.8,
) -> np.ndarray:
    """
    Build a tapered, 5-point clock-hand polygon.

    Geometry (all in pixels at the canvas scale):
        - pivot at (cx, cy)
        - widest point at 12 % of tip_len ahead of pivot
        - tapers symmetrically to a near-point at the tip
        - short tail (counterweight) behind pivot, narrower than the body

    Returns an (N, 2) float32 array of (x, y) vertices.
    """
    ar  = math.radians(angle_deg)
    fdx, fdy = math.sin(ar), -math.cos(ar)   # forward (toward tip)
    pdx, pdy = math.cos(ar),  math.sin(ar)   # perpendicular (left side)

    # Widest cross-section sits slightly ahead of the pivot
    wf    = 0.12
    wide_x = cx + wf * tip_len * fdx
    wide_y = cy + wf * tip_len * fdy

    tail_hw = max(tip_hw, base_hw * 0.30)   # tail is narrower than body

    pts = np.array([
        # tail tip (behind pivot) - narrows to a small blunt end
        [cx - tail_len * fdx + tail_hw * pdx,   cy - tail_len * fdy + tail_hw * pdy],
        [cx - tail_len * fdx - tail_hw * pdx,   cy - tail_len * fdy - tail_hw * pdy],
        # wide shoulders (near pivot)
        [wide_x - base_hw * pdx,                wide_y - base_hw * pdy],
        # tip (forward end) - near-zero width
        [cx + tip_len * fdx - tip_hw * pdx,     cy + tip_len * fdy - tip_hw * pdy],
        [cx + tip_len * fdx + tip_hw * pdx,     cy + tip_len * fdy + tip_hw * pdy],
        [wide_x + base_hw * pdx,                wide_y + base_hw * pdy],
    ], dtype=np.float32)

    return pts


def _draw_hand_ssaa(
    canvas: np.ndarray,
    cx: int, cy: int,
    angle_deg: float,
    radius: int,
    tip_ratio: float,
    tail_ratio: float,
    hw_ratio: float,
    color_bgr: tuple,
    sf: int,
    shadow: bool = True,
    shadow_opacity: float = 0.28,
) -> None:
    """
    Draw one tapered hand onto *canvas* (in-place, BGR uint8).

    Parameters
    ----------
    canvas       : BGR image to draw onto (already at super-sampled resolution)
    cx, cy       : pivot centre in super-sampled coordinates
    angle_deg    : hand angle, 0 = 12 o'clock, clockwise positive
    radius       : clock radius in super-sampled pixels
    tip_ratio    : tip length / radius
    tail_ratio   : tail (counterweight) length / radius
    hw_ratio     : body half-width / radius
    color_bgr    : (B, G, R) hand colour
    sf           : super-sampling factor (used to scale shadow offset)
    shadow       : whether to draw a drop shadow
    shadow_opacity: alpha of the drop shadow layer
    """
    tip_len  = radius * tip_ratio
    tail_len = radius * tail_ratio
    base_hw  = radius * hw_ratio
    tip_hw   = max(sf * 0.6, base_hw * 0.07)   # very narrow at tip

    pts = _make_hand_polygon(cx, cy, angle_deg, tip_len, tail_len, base_hw, tip_hw)
    pts_i = pts.astype(np.int32)

    if shadow:
        offset = max(sf, 2)
        shadow_pts = (pts + np.array([offset, offset])).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [shadow_pts], (20, 20, 20))
        cv2.addWeighted(overlay, shadow_opacity, canvas, 1.0 - shadow_opacity, 0, canvas)

    cv2.fillPoly(canvas, [pts_i], color_bgr)
    # Smooth outer edge with a thin polyline in the same colour
    cv2.polylines(canvas, [pts_i], isClosed=True, color=color_bgr, thickness=max(1, sf // 2))


def render_clock_hands(
    image: np.ndarray,
    hour_angle: float,
    minute_angle: float,
    second_angle: float,
    center: Tuple[int, int] = None,
    radius: int = None,
    # --- geometry ratios (relative to clock radius) ---
    hour_hand_length: float = 0.50,
    minute_hand_length: float = 0.72,
    second_hand_length: float = 0.87,
    # --- fallback pixel widths (overridden when source_image is given) ---
    hour_hand_width: int = 6,
    minute_hand_width: int = 4,
    second_hand_width: int = 2,
    # --- fallback colours (overridden when source_image is given) ---
    hour_hand_color: Tuple[int, int, int] = (30, 30, 30),
    minute_hand_color: Tuple[int, int, int] = (30, 30, 30),
    second_hand_color: Tuple[int, int, int] = (210, 10, 10),
    # --- optional source image for adaptive styling ---
    source_image: np.ndarray = None,
) -> np.ndarray:
    """
    Senior-level clock-hand renderer.

    The incoming *image* should already have the original hands erased
    (e.g. via erase_hands_with_face_color).  This function draws fresh,
    high-quality hands from scratch using:

      • 4× Super-Sampling Anti-Aliasing (SSAA) - no jagged edges
      • Tapered polygon hands - wide at pivot, sharp pointed tip
      • Drop shadow - adds perceived depth
      • Counterweight tail on all hands
      • Hub-cap circle on top of all hands (like a real clock screw)
      • Adaptive style - colours and widths sampled from source_image

    Parameters
    ----------
    image        : RGB ndarray with hands already removed (clean clock face)
    *_angle      : hand angles in degrees (0 = 12 o'clock, clockwise)
    source_image : original (un-erased) RGB clock image for style sampling
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        rough = min(h, w) // 2 - 10
        # Auto-detect the true inner face radius (handles thick-bordered clocks)
        radius = _detect_inner_radius(image, center, rough)
    cx, cy = center

    # ── 1. Adapt style from original clock image ──────────────────────────
    if source_image is not None:
        # Also auto-detect radius from source for style sampling
        rough_src = min(source_image.shape[0], source_image.shape[1]) // 2 - 10
        src_radius = _detect_inner_radius(source_image, center, rough_src)
        main_col, sec_col, _ = _sample_hand_style(source_image, center, src_radius)
        hour_hand_color   = main_col
        minute_hand_color = main_col
        second_hand_color = sec_col
        # Scale widths proportionally to the clock radius
        hour_hand_width   = max(4, int(radius * 0.055))
        minute_hand_width = max(3, int(radius * 0.038))
        second_hand_width = max(1, int(radius * 0.016))

    # ── 2. Prepare canvas (BGR, already-erased face) ─────────────────────
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ── 3. Super-sample: upscale 4× for anti-aliased rendering ───────────
    SF   = 4
    bh   = h  * SF
    bw   = w  * SF
    bcx  = cx * SF
    bcy  = cy * SF
    br   = radius * SF

    big  = cv2.resize(canvas, (bw, bh), interpolation=cv2.INTER_CUBIC)

    def _bgr(rgb):
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    # Half-width ratios (pixels / radius) at super-sampled scale
    h_hw = (hour_hand_width   / 2.0) / radius
    m_hw = (minute_hand_width / 2.0) / radius
    s_hw = (second_hand_width / 2.0) / radius

    # ── 4. Draw hands: hour → minute → second  (each with its own shadow) ─
    _draw_hand_ssaa(
        big, bcx, bcy, hour_angle, br,
        tip_ratio   = hour_hand_length,
        tail_ratio  = 0.10,
        hw_ratio    = h_hw,
        color_bgr   = _bgr(hour_hand_color),
        sf          = SF,
    )
    _draw_hand_ssaa(
        big, bcx, bcy, minute_angle, br,
        tip_ratio   = minute_hand_length,
        tail_ratio  = 0.13,
        hw_ratio    = m_hw,
        color_bgr   = _bgr(minute_hand_color),
        sf          = SF,
    )
    _draw_hand_ssaa(
        big, bcx, bcy, second_angle, br,
        tip_ratio   = second_hand_length,
        tail_ratio  = 0.22,          # longer counterweight on second hand
        hw_ratio    = s_hw,
        color_bgr   = _bgr(second_hand_color),
        sf          = SF,
        shadow_opacity = 0.18,       # lighter shadow for the thin second hand
    )

    # ── 5. Hub cap (sits on top of all hands, like a real clock screw) ───
    hub_r     = max(SF * 3, int(br * 0.038))
    hub_col   = _bgr(hour_hand_color)
    outer_col = (max(0, hub_col[0] - 40), max(0, hub_col[1] - 40), max(0, hub_col[2] - 40))

    cv2.circle(big, (bcx, bcy), hub_r + SF + 1, outer_col,   -1)   # dark border
    cv2.circle(big, (bcx, bcy), hub_r,           hub_col,     -1)   # main cap
    # small specular glint
    glint_r = max(1, hub_r // 3)
    cv2.circle(big, (bcx - hub_r // 3, bcy - hub_r // 3), glint_r, (220, 220, 220), -1)

    # ── 6. Downsample back to original resolution ─────────────────────────
    #    INTER_AREA is ideal for downscaling: equivalent to box-filter AA.
    result = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Legacy / convenience wrappers
# ---------------------------------------------------------------------------

def remove_clock_hands(
    image: np.ndarray,
    center: Tuple[int, int] = None,
    radius: int = None,
    inpaint_radius: int = None,
) -> np.ndarray:
    """Legacy simple hand removal. Prefer erase_hands_with_hough."""
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 2 - 20
    if inpaint_radius is None:
        inpaint_radius = int(radius * 0.35)
    mask        = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, inpaint_radius, 255, -1)
    img_inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB)
