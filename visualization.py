"""
Visualization utilities for face recognition annotations.
"""

from typing import List, Optional

import cv2
import numpy as np

from tracker import Detection


def draw_annotations(frame: np.ndarray, detections: List[Detection], fps: Optional[float] = None) -> np.ndarray:
    """
    Draw bounding boxes, labels, and FPS on frame

    Args:
        frame: BGR image (OpenCV format)
        detections: List of Detection objects
        fps: Optional FPS value to display

    Returns:
        Annotated frame (BGR)
    """
    annotated = frame.copy()

    for det in detections:
        x, y, w, h = det.bbox

        # Color: green for known, red for unknown
        color = (0, 255, 0) if det.label != "Unknown" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label_text = det.label
        cv2.putText(annotated, label_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw confidence (if known)
        if det.label != "Unknown":
            conf_text = f"{det.confidence:.2f}"
            cv2.putText(annotated, conf_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw FPS
    if fps is not None:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return annotated
