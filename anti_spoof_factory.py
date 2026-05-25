"""
Face anti-spoofing abstraction layer.

Provides a unified interface for liveness / presentation-attack detection
via the AntiSpoofChecker Protocol and a create_anti_spoof() factory function.

Design mirrors detector_factory.py / embedding_factory.py:
  Protocol  ->  concrete classes  ->  factory function

Usage:
    checker = create_anti_spoof("minifas")
    is_real, score = checker.check(face_crop)   # True/False, confidence

    checker = create_anti_spoof("none")         # disabled — returns None
    if checker is not None:
        is_real, score = checker.check(face_crop)
"""

from __future__ import annotations

import os
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Protocol (interface contract)
# ---------------------------------------------------------------------------

class AntiSpoofChecker(Protocol):
    """
    Any anti-spoof checker must implement check().

    check() receives a face crop (RGB uint8 numpy array) and returns
    (is_real, confidence) where confidence is in [0, 1].
    """

    def check(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Determine whether a face image is real or a spoof (photo/screen).

        Args:
            face_image: RGB uint8 numpy array — the detected face crop.

        Returns:
            (is_real, confidence): is_real is True for live faces,
            confidence is the model's certainty in [0, 1].
        """
        ...


# ---------------------------------------------------------------------------
# MiniFASNet checker (MiniFASNetV2-SE via ONNX Runtime)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "anti_spoof.onnx")


class MiniFASChecker:
    """
    MiniFASNetV2-SE anti-spoofing model (ONNX, ~600 KB quantized).

    Classifies face crops as real or spoof (print / screen replay attacks).
    Trained on CelebA-Spoof (~70k samples), reported 98.2% accuracy.

    Input: 128x128 RGB face crop (aspect-ratio preserved, reflection padded).
    Output: 2 logits [real, spoof]. real_logit >= spoof_logit => live face.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            model_path: Path to the anti-spoof ONNX model.
            threshold: Classification threshold in [0, 1].
                       0.5 = balanced (default).
                       Higher = stricter (fewer false accepts, more false rejects).
        """
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Anti-spoof model not found at {model_path}. "
                "Download best_model_quantized.onnx from "
                "https://github.com/SuriAI/face-antispoof-onnx"
            )

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._img_size = 128

        # Convert probability threshold to logit-space threshold.
        # At p=0.5 => logit_threshold=0.0 (real_logit >= spoof_logit).
        p = max(1e-6, min(1 - 1e-6, threshold))
        self._logit_threshold = float(np.log(p / (1 - p)))

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Resize with aspect-ratio preservation + reflection padding to 128x128,
        then normalize to [0, 1] float32 CHW.
        """
        size = self._img_size

        # Aspect-ratio-preserving resize (123 to 138)
        h, w = face_image.shape[:2]
        ratio = float(size) / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # this below picks the resize algo. 
        # INTER_LANCZSO4 is better for upscaling and making images sharper
        # INTER_AREA is better for downscaling and avoids aliasing/jaggedness
        interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        img = cv2.resize(face_image, (new_w, new_h), interpolation=interp)

        # Reflection padding to fill 128x128
        delta_h = size - new_h
        delta_w = size - new_w
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        # HWC -> CHW, float32, normalize to [0, 1]
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img

    def check(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Classify a face crop as real or spoof.

        Args:
            face_image: RGB uint8 numpy array (any size — will be resized).

        Returns:
            (is_real, confidence):
                is_real: True if the face appears live.
                confidence: model certainty in [0, 1], where 1.0 = very certain.
        """
        # Preprocess and add batch dimension
        blob = self._preprocess(face_image)
        blob = np.expand_dims(blob, axis=0)  # (1, 3, 128, 128)

        # Run inference
        logits = self._session.run(None, {self._input_name: blob})[0]  # (1, 2)
        real_logit = float(logits[0][0])
        spoof_logit = float(logits[0][1])

        logit_diff = real_logit - spoof_logit
        is_real = logit_diff >= self._logit_threshold

        # Convert logit diff to a 0-1 confidence via sigmoid
        confidence = 1.0 / (1.0 + np.exp(-abs(logit_diff)))

        return bool(is_real), float(confidence)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_anti_spoof(method: str = "none", **kwargs) -> Optional[AntiSpoofChecker]:
    """
    Instantiate an anti-spoof checker by name.

    Args:
        method: "none" (disabled) or "minifas" (MiniFASNetV2-SE ONNX).
        **kwargs: Passed to the checker constructor.
                  MiniFASChecker: model_path (optional), threshold (optional)

    Returns:
        An AntiSpoofChecker instance, or None if method="none".
        Callers must guard with `if checker is not None` before calling .check().
    """
    if method == "none":
        return None

    elif method == "minifas":
        return MiniFASChecker(**kwargs)

    else:
        raise ValueError(
            f"Unknown anti-spoof method: '{method}'. "
            "Choose 'none' or 'minifas'."
        )
