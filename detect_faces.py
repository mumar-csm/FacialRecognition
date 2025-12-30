""" Script strictly for face detection using Haar Cascades """

import argparse
import time
import sys
from pathlib import Path

import cv2

def load_haar_cascade():
    # Uses OpenCV's built-in path to haar cascades
    cv2.setUseOptimized(True)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    if not cascade_path.exists():
        print(f"[ERROR] Haar cascade file not found at {cascade_path}!")
        sys.exit(1)

    cascade = cv2.CascadeClassifier(str(cascade_path))

    if cascade.empty():
        print(f"[ERROR] Failed to load Haar cascade from {cascade_path}!")
        sys.exit(1)
    return cascade

def detect_faces_in_image(image_path: Path, scale_factor=1.1, min_neighbors=5, min_size=(60,60)):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        sys.exit(1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = load_haar_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    # Draw rectangles around detected faces, specifically x axis and y axis with width and height
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    print(f"[INFO] Detected {len(faces)} face(s) in {image_path.name}.")
    cv2.imshow("Faces (Image)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_from_webcam(camera_index=0, scale_factor=1.1, min_neighbors=5, min_size=(60,60), resize_width=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {camera_index}!")
        sys.exit(1)

    cascade = load_haar_cascade()
    prev_time = time.time()
    fps_smooth = None
    print("[INFO] Starting webcam face detection. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam!")
                break
            
            if resize_width and resize_width > 0 and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # FPS estimation
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else(0.9 * fps_smooth + 0.1 * fps)

            # Adding FPS display on frame
            cv2.putText(frame, f"FPS: {fps_smooth:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Faces (Webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Face detection using OpenCV Haar Cascades (image or webcam).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file.")
    group.add_argument("--webcam", action="store_true", help="Use webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, nargs=2, default=(60, 60), metavar=("W", "H"))
    parser.add_argument("--resize-width", type=int, default=None, help="Downscale frame width for speed (e.g., 640).")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.image:
        detect_faces_in_image(Path(args.image), args.scale_factor, args.min_neighbors, tuple(args.min_size))
    else:
        detect_faces_from_webcam(args.camera_index, args.scale_factor, args.min_neighbors, tuple(args.min_size), args.resize_width)

if __name__ == "__main__":
    main()