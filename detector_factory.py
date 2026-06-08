"""
Face detector abstraction layer.

Provides a unified interface for multiple detection backends (Haar, RetinaFace)
via the FaceDetector Protocol and a create_detector() factory function.

Also includes align_face() for landmark-based face alignment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Protocol (interface contract)
# ---------------------------------------------------------------------------

class FaceDetector(Protocol):
    """
    Any face detector must implement detect().

    Returns a list of (bbox, landmarks) tuples:
      - bbox: (x, y, w, h) in pixel coordinates
      - landmarks: 5x2 numpy array of facial landmarks, or None if unavailable
                   Order: left_eye, right_eye, nose, left_mouth, right_mouth
    """

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        ...


# ---------------------------------------------------------------------------
# Haar Cascade detector (legacy, fast, no landmarks)
# ---------------------------------------------------------------------------

class HaarDetector:
    """
    OpenCV Haar Cascade face detector.

    Fast (~5-10ms) but lower accuracy than deep-learning detectors.
    Does NOT produce landmarks, so align_face() won't work with this backend.
    """

    def __init__(self, cascade_path: str, params: Optional[Dict[str, Any]] = None):
        if not os.path.isfile(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")

        self._params = params or {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": (60, 60),
        }

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """Detect faces via Haar cascade. Returns (bbox, None) — no landmarks."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._params.get("scale_factor", 1.1),
            minNeighbors=self._params.get("min_neighbors", 5),
            minSize=self._params.get("min_size", (60, 60)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return [(tuple(int(v) for v in box), None) for box in boxes]


# ---------------------------------------------------------------------------
# RetinaFace detector (InsightFace, accurate, provides landmarks)
# ---------------------------------------------------------------------------

class RetinaFaceDetector:
    """
    InsightFace RetinaFace detector.

    Higher accuracy than Haar, returns 5-point landmarks for alignment.
    Runs on CPU via ONNX Runtime by default; GPU-ready with provider swap.

    First call downloads the model (~100 MB) to ~/.insightface/models/.
    """

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            det_size: Input resolution for the detector network.
                      Larger = more accurate but slower.
                      (640, 640) is a good balance for CPU.
        """
        import warnings
        warnings.filterwarnings("ignore", message=".*CUDA.*not available.*|.*CUDAExecutionProvider.*")
        import insightface  # lazy import — only loaded when selected

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """
        Detect faces with RetinaFace.

        InsightFace expects BGR input, so we convert from RGB here.
        Returns (bbox, landmarks) where landmarks is a 5x2 float array.
        """ 
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self._app.get(bgr)

        results = []
        for face in faces:
            # InsightFace bbox is [x1, y1, x2, y2] as float
            x1, y1, x2, y2 = face.bbox.astype(int)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 5-point landmarks (5x2 array), may be None
            landmarks = face.kps if face.kps is not None else None

            results.append((bbox, landmarks))

        return results


# ---------------------------------------------------------------------------
# SCRFD detector (InsightFace, direct model_zoo load — no auxiliary models)
# ---------------------------------------------------------------------------

class SCRFDDetector:
    """
    InsightFace SCRFD detector loaded directly via model_zoo.

    The RetinaFaceDetector above goes through insightface.app.FaceAnalysis,
    which runs every model in the pack (2d106det, 1k3d68, genderage) on every
    frame even though we only need the 5-point landmarks the detector itself
    produces. Loading via model_zoo skips that overhead — same detection
    model, same accuracy, just no auxiliary pipeline.
    """

    def __init__(self, model_path: str,
                 det_size: Tuple[int, int] = (320, 320),
                 det_thresh: float = 0.5):
        model_path = os.path.expanduser(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        # Silence ONNX runtime "VerifyOutputSizes" warnings. The SCRFD ONNX was
        # exported assuming 640x640 input; at 320x320 the actual output shapes
        # are smaller than the metadata declares, which is harmless but floods
        # stderr on every call. 3 = ERROR-only.
        import onnxruntime as ort
        ort.set_default_logger_severity(3)

        from insightface.model_zoo import model_zoo  # lazy import
        self._det = model_zoo.get_model(model_path)
        # ctx_id=-1 → CPU, ctx_id=0 → GPU
        self._det.prepare(ctx_id=-1, input_size=det_size, det_thresh=det_thresh)

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """Detect faces with SCRFD. Returns (bbox, landmarks) per face."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bboxes, kpss = self._det.detect(bgr, max_num=0, metric="default")

        results = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            landmarks = kpss[i] if kpss is not None else None
            results.append((box, landmarks))
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_detector(detector_type: str, **kwargs) -> FaceDetector:
    """
    Instantiate a face detector by name.

    Args:
        detector_type: "haar", "retinaface", or "scrfd"
        **kwargs: Passed to the detector constructor.
                  HaarDetector: cascade_path (required), params (optional)
                  RetinaFaceDetector: det_size (optional, default (640,640))
                  SCRFDDetector: model_path (required), det_size, det_thresh

    Returns:
        A FaceDetector instance.
    """
    if detector_type == "haar":
        cascade_path = kwargs.get("cascade_path")
        if not cascade_path:
            raise ValueError("HaarDetector requires 'cascade_path' kwarg")
        params = kwargs.get("params")
        return HaarDetector(cascade_path, params)

    elif detector_type == "retinaface":
        det_size = kwargs.get("det_size", (640, 640))
        return RetinaFaceDetector(det_size=det_size)

    elif detector_type == "scrfd":
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("SCRFDDetector requires 'model_path' kwarg")
        det_size = kwargs.get("det_size", (320, 320))
        det_thresh = kwargs.get("det_thresh", 0.5)
        return SCRFDDetector(model_path, det_size=det_size, det_thresh=det_thresh)

    else:
        raise ValueError(f"Unknown detector type: '{detector_type}'. Choose 'haar', 'retinaface', or 'scrfd'.")


# ---------------------------------------------------------------------------
# Face alignment utility
# ---------------------------------------------------------------------------

# Standard reference landmarks for a 112x112 aligned face.
# These are the canonical positions used by ArcFace training.
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float64)


def align_face(image: np.ndarray,
               landmarks: np.ndarray,
               output_size: int = 112) -> np.ndarray:
    """
    Warp a face region into a canonical aligned image using 5-point landmarks.

    Uses a similarity transform (rotation + uniform scale + translation) so
    facial geometry is preserved. This is the standard alignment used by
    ArcFace and most modern face recognition models.

    Args:
        image: Full frame (RGB, numpy array).
        landmarks: 5x2 array of detected landmarks.
        output_size: Output square size in pixels (default 112 for ArcFace).

    Returns:
        Aligned face crop (output_size x output_size, RGB, uint8).
    """
    from skimage.transform import SimilarityTransform

    # Scale reference landmarks if output size differs from 112
    if output_size != 112:
        scale = output_size / 112.0
        ref = REFERENCE_LANDMARKS_112 * scale
    else:
        ref = REFERENCE_LANDMARKS_112

    # Estimate similarity transform: landmarks -> reference
    tform = SimilarityTransform.from_estimate(landmarks.astype(np.float64), ref)

    # Build 2x3 affine matrix for cv2.warpAffine
    M = tform.params[:2]

    aligned = cv2.warpAffine(
        image, M,
        (output_size, output_size),
        borderValue=(0, 0, 0),
    )

    return aligned
