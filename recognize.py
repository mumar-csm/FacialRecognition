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

  # Video file (not yet implemented)
  python recognize.py --mode video --source meeting.mp4 --output annotated.mp4
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
            # Video mode (not yet implemented)
            print("[ERROR] Video mode not yet implemented")
            print("[INFO] Use webcam mode for now or process video frames manually")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
