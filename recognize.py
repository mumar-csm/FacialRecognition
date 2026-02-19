#!/usr/bin/env python3
"""
Runtime face recognition system
Identifies employees from webcam/video feeds or static images using pre-built face database
"""

import os
import sys
import pickle
import argparse
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import face_recognition

# Import distance calculation functions
from euclideanDist import euclidean_distance, l2_normalize

# Import EncodingsDB dataclass from build_encodings
from build_encodings import EncodingsDB


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
        Determine if we need to run expensive encoding.

        Returns True if:
        - Interval reached (e.g., 30 frames)
        - Different number of faces
        - Faces moved significantly (IoU < 0.5)
        """
        self.frames_since_identify += 1

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

        return False  # All faces stable, reuse cache

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

        Strategy: Match current boxes to previous boxes by IoU,
        reuse the label from the best matching previous detection.
        """
        detections = []

        for curr_box in current_boxes:
            # Find best matching previous box
            best_iou = 0.0
            best_idx = 0

            for i, prev_box in enumerate(self.last_boxes):
                iou = self.compute_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # Use cached identity from best match
            if best_iou > 0.3:  # Reasonable overlap
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


def load_database(db_path: str) -> Tuple[List[List[float]], List[str]]:
    """
    Load face database from pickle file

    Args:
        db_path: Path to known_faces.pkl

    Returns:
        Tuple of (encodings_list, labels_list)

    Raises:
        FileNotFoundError: If database doesn't exist
        Exception: If database is corrupted or invalid
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    try:
        with open(db_path, "rb") as f:
            db = pickle.load(f)

        if not isinstance(db, EncodingsDB):
            raise ValueError("Invalid database format")

        if len(db.encodings) == 0:
            raise ValueError("Database is empty")

        print(f"[INFO] Loaded database: {len(db.encodings)} employees")
        print(f"[INFO] Employees: {', '.join(db.labels)}")

        return db.encodings, db.labels

    except Exception as e:
        raise Exception(f"Failed to load database: {e}")


def find_best_match(unknown_encoding: np.ndarray,
                   known_encodings: List[List[float]],
                   labels: List[str],
                   threshold: float = 1.0) -> Tuple[str, float, float]:
    """
    Find best matching face from database

    Args:
        unknown_encoding: 128-d face embedding to match
        known_encodings: List of known face embeddings
        labels: List of employee IDs corresponding to encodings
        threshold: Distance threshold for positive match (default: 1.0)

    Returns:
        Tuple of (label, distance, confidence)
        - If match found: (employee_id, distance, confidence)
        - If no match: ("Unknown", min_distance, 0.0)
    """
    if len(known_encodings) == 0:
        return ("Unknown", float('inf'), 0.0)

    # Calculate distances to all known faces
    distances = []
    for known_enc in known_encodings:
        known_arr = np.array(known_enc)
        dist = euclidean_distance(unknown_encoding, known_arr)
        distances.append(dist)

    # Find minimum distance
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    # Check if below threshold
    if min_dist <= threshold:
        # Match found
        label = labels[min_idx]
        confidence = 1.0 - (min_dist / threshold)  # Convert distance to confidence
        return (label, min_dist, confidence)
    else:
        # No match
        return ("Unknown", min_dist, 0.0)


def detect_and_encode_faces(frame: np.ndarray,
                            cascade_path: str,
                            detector_params: Dict[str, Any]) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Detect faces in frame and compute embeddings

    Args:
        frame: RGB image (numpy array)
        cascade_path: Path to haarcascade XML
        detector_params: Detection parameters (scale_factor, min_neighbors, min_size)

    Returns:
        List of (bbox, encoding) tuples
        bbox: (x, y, w, h)
        encoding: 128-d numpy array
    """
    # Load Haar Cascade
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    boxes = classifier.detectMultiScale(
        gray,
        scaleFactor=detector_params.get("scale_factor", 1.1),
        minNeighbors=detector_params.get("min_neighbors", 5),
        minSize=detector_params.get("min_size", (60, 60)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    results = []

    # Encode each detected face
    for (x, y, w, h) in boxes:
        # Convert Haar bbox (x,y,w,h) to face_recognition format (top, right, bottom, left)
        fr_box = (y, x + w, y + h, x)

        try:
            # Try encoding with known face location first
            encodings = face_recognition.face_encodings(frame, known_face_locations=[fr_box])

            if encodings:
                results.append(((x, y, w, h), encodings[0]))
            else:
                # Fallback: crop ROI and encode
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    face_roi = np.ascontiguousarray(face_roi)
                    encodings = face_recognition.face_encodings(face_roi)
                    if encodings:
                        results.append(((x, y, w, h), encodings[0]))

        except Exception as e:
            print(f"[WARN] Encoding failed for face at ({x},{y}): {e}")
            continue

    return results


def detect_faces_only(frame: np.ndarray,
                      cascade_path: str,
                      detector_params: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in frame WITHOUT encoding (fast).

    Used by SimpleTracker to quickly detect faces every frame,
    then only encode when needed (every N frames or on movement).

    Args:
        frame: RGB image (numpy array)
        cascade_path: Path to haarcascade XML
        detector_params: Detection parameters (scale_factor, min_neighbors, min_size)

    Returns:
        List of bounding boxes in (x, y, w, h) format
    """
    # Load Haar Cascade
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    boxes = classifier.detectMultiScale(
        gray,
        scaleFactor=detector_params.get("scale_factor", 1.1),
        minNeighbors=detector_params.get("min_neighbors", 5),
        minSize=detector_params.get("min_size", (60, 60)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return [tuple(box) for box in boxes]


def process_frame(frame: np.ndarray,
                 known_encodings: List[List[float]],
                 labels: List[str],
                 cascade_path: str,
                 threshold: float = 1.0,
                 detector_params: Optional[Dict[str, Any]] = None) -> List[Detection]:
    """
    Process a single frame: detect faces, encode, and match

    Args:
        frame: RGB image (numpy array)
        known_encodings: List of known face embeddings
        labels: List of employee IDs
        cascade_path: Path to Haar Cascade XML
        threshold: Matching threshold
        detector_params: Detection parameters

    Returns:
        List of Detection objects
    """
    if detector_params is None:
        detector_params = {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": (60, 60)
        }

    # Detect and encode faces
    face_data = detect_and_encode_faces(frame, cascade_path, detector_params)

    # Match each face
    detections = []
    for (bbox, encoding) in face_data:
        label, distance, confidence = find_best_match(encoding, known_encodings, labels, threshold)
        detection = Detection(bbox=bbox, label=label, distance=distance, confidence=confidence)
        detections.append(detection)

    return detections


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


def recognize_from_webcam(db_path: str,
                         camera_index: int = 0,
                         threshold: float = 1.0,
                         cascade_path: str = "data/haarcascade_frontalface_default.xml",
                         resize_width: int = 640,
                         fps_display: bool = True) -> None:
    """
    Real-time face recognition from webcam

    Args:
        db_path: Path to face database
        camera_index: Camera index (0 for default)
        threshold: Matching threshold
        cascade_path: Path to Haar Cascade XML
        resize_width: Resize frame width for performance
        fps_display: Show FPS counter
    """
    # Load database
    print("[INFO] Loading face database...")
    known_encodings, labels = load_database(db_path)

    # Open webcam
    print(f"[INFO] Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")

    print("[INFO] Camera opened successfully")
    print("[INFO] Press 'q' to quit")

    # FPS tracking
    fps_smooth = 0.0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                break

            # Resize for performance
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

            # Convert BGR to RGB for face_recognition
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            detections = process_frame(frame_rgb, known_encodings, labels, cascade_path, threshold)

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            if dt > 0:
                fps_instant = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

            # Draw annotations
            annotated = draw_annotations(frame, detections, fps_smooth if fps_display else None)

            # Display
            cv2.imshow("Face Recognition (Press 'q' to quit)", annotated)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit signal received")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released")


def recognize_from_image(image_path: str,
                        db_path: str,
                        threshold: float = 1.0,
                        cascade_path: str = "data/haarcascade_frontalface_default.xml",
                        output_path: Optional[str] = None,
                        display: bool = True) -> List[Detection]:
    """
    Face recognition on a single image

    Args:
        image_path: Path to input image
        db_path: Path to face database
        threshold: Matching threshold
        cascade_path: Path to Haar Cascade XML
        output_path: Optional path to save annotated image
        display: Display result window

    Returns:
        List of Detection objects
    """
    # Load database
    print("[INFO] Loading face database...")
    known_encodings, labels = load_database(db_path)

    # Load image
    print(f"[INFO] Loading image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Process frame
    print("[INFO] Processing image...")
    detections = process_frame(frame_rgb, known_encodings, labels, cascade_path, threshold)

    print(f"[INFO] Detected {len(detections)} face(s)")
    for det in detections:
        print(f"  - {det.label}: distance={det.distance:.3f}, confidence={det.confidence:.3f}")

    # Draw annotations
    annotated = draw_annotations(frame_bgr, detections)

    # Save if requested
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved result to: {output_path}")

    # Display if requested
    if display:
        cv2.imshow("Face Recognition (Press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def recognize_from_video(video_path: str,
                        db_path: str,
                        threshold: float = 1.0,
                        cascade_path: str = "data/haarcascade_frontalface_default.xml",
                        output_path: Optional[str] = None,
                        frame_skip: int = 0,
                        resize_width: int = 640,
                        display: bool = False) -> Dict[str, Any]:
    """
    Face recognition on video file

    Args:
        video_path: Path to input video file (.mp4, .avi, .mov)
        db_path: Path to face database
        threshold: Matching threshold
        cascade_path: Path to Haar Cascade XML
        output_path: Optional path to save annotated video
        frame_skip: Skip frames (0=all, 1=every other, 2=every 3rd, etc.)
        resize_width: Resize frame width for performance
        display: Show frames in real-time window during processing

    Returns:
        Dictionary with statistics:
        - total_frames: Total frames processed
        - faces_detected: Total faces detected across all frames
        - unique_identities: Set of unique identities
        - processing_time: Total processing time in seconds
    """
    # Load database
    print("[INFO] Loading face database...")
    known_encodings, labels = load_database(db_path)

    # Warn if no output and no display
    if not output_path and not display:
        print("[WARN] No output file specified and display disabled")
        print("[WARN] Video will be processed but no results will be visible")
        print("[WARN] To save output: use --output flag")
        print("[WARN] To view in real-time: use --display flag")

    # Open video
    print(f"[INFO] Attempting to open video: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video properties: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames total")
    print(f"[INFO] Frame skip: {frame_skip} (processing every {frame_skip + 1}th frame)")

    # Create video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[WARN] Failed to create video writer for {output_path}")
            writer = None
        else:
            print(f"[INFO] Will save annotated video to: {output_path}")

    # Statistics
    start_time = time.time()
    frame_idx = 0
    faces_detected = 0
    unique_identities = set()

    print("[INFO] Processing video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skip logic
            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                continue

            # Resize for performance
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

            # Convert BGR to RGB for face_recognition
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            detections = process_frame(frame_rgb, known_encodings, labels, cascade_path, threshold)

            # Update statistics
            faces_detected += len(detections)
            for det in detections:
                if det.label != "Unknown":
                    unique_identities.add(det.label)

            # Draw annotations
            annotated = draw_annotations(frame, detections)

            # Write to output video if requested
            if writer and annotated.shape[1] == width and annotated.shape[0] == height:
                writer.write(annotated)

            # Display in real-time window if requested
            if display:
                cv2.imshow("Video Processing (Press 'q' to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit signal received")
                    break

            # Progress reporting every 30 frames
            if (frame_idx + 1) % 30 == 0:
                progress = (frame_idx + 1) / max(total_frames, 1) * 100
                print(f"[INFO] Processing frame {frame_idx + 1}/{total_frames} ({progress:.1f}%)")

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Faces detected: {faces_detected}")
    print(f"  Unique identities: {len(unique_identities)}")
    print(f"  Processing time: {elapsed_time:.1f}s")
    print(f"  Average FPS: {frame_idx / elapsed_time:.1f}")

    return {
        "total_frames": frame_idx,
        "faces_detected": faces_detected,
        "unique_identities": len(unique_identities),
        "processing_time": elapsed_time
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Runtime face recognition system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam mode
  python recognize.py --mode webcam --source 0 --threshold 1.0

  # Single image
  python recognize.py --mode image --source photo.jpg --output result.jpg

  # Video file - save output
  python recognize.py --mode video --source meeting.mp4 --output annotated.mp4

  # Video file - watch in real-time
  python recognize.py --mode video --source meeting.mp4 --display

  # Video file - save and watch in real-time
  python recognize.py --mode video --source meeting.mp4 --output annotated.mp4 --display --frame-skip 1
        """
    )

    parser.add_argument("--mode", choices=["webcam", "image", "video"],
                       default="webcam",
                       help="Recognition mode (default: webcam)")

    parser.add_argument("--source",
                       help="Camera index (webcam mode) or image/video path")

    parser.add_argument("--database", default="data/known_faces.pkl",
                       help="Path to face database pickle file (default: data/known_faces.pkl)")

    parser.add_argument("--threshold", type=float, default=1.0,
                       help="Matching threshold (default: 1.0, lower=stricter)")

    parser.add_argument("--output",
                       help="Output path for saving results (image/video mode)")

    parser.add_argument("--resize-width", type=int, default=640,
                       help="Resize frame width for performance (default: 640)")

    parser.add_argument("--no-display", action="store_true",
                       help="Disable visualization window")

    parser.add_argument("--cascade", default="data/haarcascade_frontalface_default.xml",
                       help="Path to Haar Cascade XML (default: data/haarcascade_frontalface_default.xml)")

    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Skip frames (0=all, 1=every other, 2=every 3rd, etc.) (default: 0)")

    parser.add_argument("--display", action="store_true",
                       help="Display video frames in real-time window during processing (video mode)")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    try:
        if args.mode == "webcam":
            # Webcam mode
            camera_index = int(args.source) if args.source else 0
            recognize_from_webcam(
                db_path=args.database,
                camera_index=camera_index,
                threshold=args.threshold,
                cascade_path=args.cascade,
                resize_width=args.resize_width,
                fps_display=not args.no_display
            )

        elif args.mode == "image":
            # Image mode
            if not args.source:
                print("[ERROR] --source required for image mode")
                sys.exit(1)

            recognize_from_image(
                image_path=args.source,
                db_path=args.database,
                threshold=args.threshold,
                cascade_path=args.cascade,
                output_path=args.output,
                display=not args.no_display
            )

        elif args.mode == "video":
            # Video file mode
            if not args.source:
                print("[ERROR] --source required for video mode")
                sys.exit(1)

            recognize_from_video(
                video_path=args.source,
                db_path=args.database,
                threshold=args.threshold,
                cascade_path=args.cascade,
                output_path=args.output,
                frame_skip=args.frame_skip,
                resize_width=args.resize_width,
                display=args.display
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
