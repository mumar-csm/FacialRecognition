"""
Face tracking and detection result types.

SimpleTracker reduces encoding overhead by detecting faces every frame (fast)
but only encoding when needed (expensive). Re-identifies when faces move,
count changes, or interval elapses.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_DETECTOR_PARAMS = {
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": (60, 60),
}


# Detection result dataclass
@dataclass
class Detection:
    """Single face detection result"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str
    distance: float
    confidence: float


class SimpleTracker:
    """
    Intelligent face tracking to reduce encoding overhead.

    Strategy:
    - Detect faces EVERY frame (Haar is fast: 5-10ms)
    - Encode faces ONLY when needed (expensive: 100-200ms)
    - Re-identify when:
      1. N frames elapsed (default: 30 frames = 1 second)
      2. Faces moved significantly (IoU < 0.5)
      3. Number of faces changed

    Performance: 3-5x speedup (5 FPS → 15-25 FPS)
    """

    def __init__(self, reidentify_interval: int = 30):
        self.reidentify_interval = reidentify_interval
        self.last_boxes: List[Tuple[int, int, int, int]] = []
        self.last_labels: List[str] = []
        self.last_confidences: List[float] = []
        self.last_distances: List[float] = []
        self.frames_since_identify = 0
        # Cascade caching to avoid reloading every frame
        self._classifier: Optional[cv2.CascadeClassifier] = None
        self._cascade_path: Optional[str] = None

    def _get_classifier(self, cascade_path: str) -> cv2.CascadeClassifier:
        """Load cascade once, reuse on subsequent calls."""
        if self._classifier is None or self._cascade_path != cascade_path:
            if not os.path.isfile(cascade_path):
                raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
            self._classifier = cv2.CascadeClassifier(cascade_path)
            if self._classifier.empty():
                raise RuntimeError(f"Failed to load cascade: {cascade_path}")
            self._cascade_path = cascade_path
        return self._classifier

    def compute_iou(self, box1: Tuple[int, int, int, int],
                    box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.

        Args:
            box1, box2: (x, y, w, h) format

        Returns:
            IoU ratio (0.0 to 1.0)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to (x1, y1, x2, y2) format
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        # Intersection rectangle
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        # Check if there's intersection
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        # Calculate areas
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def should_reidentify(self, current_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """
        Determine if we need to run expensive encoding in the current frame.

        Returns True if:
        - Interval reached (e.g., 30 frames)
        - Different number of faces
        - Faces moved significantly (IoU < 0.5)
        """
        # Force re-identify every N frames
        if self.frames_since_identify >= self.reidentify_interval:
            return True

        # No previous data - must identify
        if len(self.last_boxes) == 0:
            return True

        # Different number of faces - must re-identify
        if len(current_boxes) != len(self.last_boxes):
            return True

        # Check if faces moved significantly
        for curr_box in current_boxes:
            max_iou = 0.0
            for prev_box in self.last_boxes:
                iou = self.compute_iou(curr_box, prev_box)
                max_iou = max(max_iou, iou)

            # If any face moved significantly, re-identify all
            if max_iou < 0.5:
                return True

        # All faces stable - increment counter and reuse cache
        self.frames_since_identify += 1
        return False

    def update(self, boxes: List[Tuple[int, int, int, int]],
               labels: List[str],
               confidences: List[float],
               distances: List[float]) -> None:
        """Store current frame data as cache."""
        self.last_boxes = boxes.copy()
        self.last_labels = labels.copy()
        self.last_confidences = confidences.copy()
        self.last_distances = distances.copy()
        self.frames_since_identify = 0  # Reset counter

    def get_cached_detections(self, current_boxes: List[Tuple[int, int, int, int]]) -> List[Detection]:
        """
        Map cached identities to current bounding boxes.

        Uses greedy assignment to prevent duplicate identity matches.
        """
        detections = []
        used_indices: set = set()  # Track which previous boxes have been assigned

        for curr_box in current_boxes:
            # Find best matching previous box (that hasn't been used)
            best_iou = 0.0
            best_idx = -1

            for i, prev_box in enumerate(self.last_boxes):
                if i in used_indices:  # Skip already-assigned boxes
                    continue
                iou = self.compute_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # Use cached identity from best match
            if best_idx >= 0 and best_iou > 0.3:  # Reasonable overlap
                used_indices.add(best_idx)  # Mark as used
                detection = Detection(
                    bbox=curr_box,
                    label=self.last_labels[best_idx],
                    distance=self.last_distances[best_idx],
                    confidence=self.last_confidences[best_idx]
                )
            else:
                # No good match - mark as unknown
                detection = Detection(
                    bbox=curr_box,
                    label="Unknown",
                    distance=999.0,
                    confidence=0.0
                )

            detections.append(detection)

        return detections

    def detect_faces(self, frame: np.ndarray, cascade_path: str,
                     detector_params: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using cached classifier (fast, no encoding).

        Args:
            frame: RGB image (numpy array)
            cascade_path: Path to haarcascade XML
            detector_params: Detection parameters (scale_factor, min_neighbors, min_size)

        Returns:
            List of bounding boxes in (x, y, w, h) format
        """
        classifier = self._get_classifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        boxes = classifier.detectMultiScale(
            gray,
            scaleFactor=detector_params.get("scale_factor", 1.1),
            minNeighbors=detector_params.get("min_neighbors", 5),
            minSize=detector_params.get("min_size", (60, 60)),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [tuple(box) for box in boxes]
