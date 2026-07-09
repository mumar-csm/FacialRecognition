"""
Anti-spoof input conditioning + lighting quality gate.

Extracted from kiosk_server.py so the spoof-crop path can be reused (e.g. by
tools/capture_spoof_samples.py) WITHOUT importing kiosk_server, whose module
body parses argv and loads models at import time.

Single source of truth: kiosk_server imports these; do not fork the logic.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# Backlit / white-background faces get underexposed by camera auto-exposure and
# read as spoofs to MiniFAS. We normalize the crop's illumination before the
# check, and — when a face is still rejected but the crop is genuinely poorly lit
# — report it as a recoverable lighting problem rather than a presentation attack.
FACE_LUMA_MIN = 60.0      # mean luma below this = underexposed face
FACE_LUMA_MAX = 225.0     # mean luma above this = blown-out face
FACE_CONTRAST_MIN = 18.0  # std below this = flat / washed-out crop
SPOOF_CROP_INSET = 0.12   # trim each bbox side toward center before MiniFAS


def _inset_bbox(x: int, y: int, w: int, h: int, frac: float, shape) -> Tuple[int, int, int, int]:
    """Shrink a bbox toward its center by `frac` per side, clamped to the frame.

    Drops the bright background corners a raw detector bbox includes (which
    MiniFAS can mistake for a screen bezel) without switching to full ArcFace
    alignment, whose tighter crop scale the model was not tuned on.
    """
    dx, dy = int(w * frac), int(h * frac)
    nx, ny, nw, nh = x + dx, y + dy, w - 2 * dx, h - 2 * dy
    if nw <= 0 or nh <= 0:
        return x, y, w, h  # degenerate — fall back to the original bbox
    H, W = shape[:2]
    nx, ny = max(0, nx), max(0, ny)
    return nx, ny, min(nw, W - nx), min(nh, H - ny)


def normalize_face_illumination(crop_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the luma channel to restore contrast/exposure on a dark face.

    Applied only to the MiniFAS input — recognition embeds the separately-aligned
    crop, so this cannot affect match accuracy.
    """
    if crop_rgb.size == 0:
        return crop_rgb
    y, cr, cb = cv2.split(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb))
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2RGB)


def assess_face_lighting(crop_rgb: np.ndarray) -> Tuple[bool, str]:
    """Judge whether a raw face crop is well-enough exposed to trust a spoof
    verdict. Runs on the un-normalized crop so it reflects the true capture.
    Returns (ok, reason); reason is empty when ok.
    """
    if crop_rgb.size == 0:
        return True, ""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    mean, std = float(np.mean(gray)), float(np.std(gray))
    if mean < FACE_LUMA_MIN:
        return False, "too_dark"
    if mean > FACE_LUMA_MAX:
        return False, "too_bright"
    if std < FACE_CONTRAST_MIN:
        return False, "low_contrast"
    return True, ""


def evaluate_anti_spoof(state, frame_rgb: np.ndarray, bbox) -> Tuple[bool, float, bool]:
    """Run MiniFAS on an illumination-normalized, background-trimmed face crop.

    Returns (is_real, score, lighting_ok). lighting_ok is only meaningful when
    is_real is False: it tells the caller whether the rejection is likely a
    lighting problem (recoverable — guide the user) vs a genuine spoof.
    """
    x, y, w, h = _inset_bbox(*bbox, SPOOF_CROP_INSET, frame_rgb.shape)
    raw = frame_rgb[y:y+h, x:x+w]
    if raw.size == 0:
        return True, 1.0, True  # nothing to judge — don't block on an empty crop
    is_real, score = state.anti_spoof.check(normalize_face_illumination(raw))
    lighting_ok = True if is_real else assess_face_lighting(raw)[0]
    return is_real, score, lighting_ok
