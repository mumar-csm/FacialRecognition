# build_encodings.py

"""
Purpose: Precompute and package employee face embeddings into a single, fast-to-load file for runtime recognition.
Value: Speeds up recognition, isolates data prep from inference, improves reliability, and sets you up to scale cleanly.
Lifecycle: Collect photos → build encodings (.pkl) → recognize at runtime → repeat as your dataset improves.
"""
from __future__ import annotations
import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np
import face_recognition
import pickle
import hashlib
from datetime import datetime
import face_recognition

#  Data models/schemas

@dataclass(frozen=True)
class ImageRecord:
    """Represents a discovered image and its derived identity."""
    employee_id: str
    image_path: str

@dataclass
class FaceRecord:
    """Represents a single encoded face and minimal metadata."""
    label: str # employee_id
    encoding: List[float] # 128-d face encoding from face_recognition/dlib
    image_path: str
    box: Tuple[int, int, int, int] # (top, right, bottom, left)/(x,y,width,height)

@dataclass
class EncodingsDB:
    """
    Serialized dataset schema for known faces.
    Keep it simple for MVP; expandable later.
    """
    encodings: List[List[float]]
    labels: List[str]
    meta: List[Dict[str, Any]] # per-record metadata (image_path, box, etc.)
    version: str = "schema_v1"


# Config

def load_config() -> Dict[str, Any]:
    """
    Return a small, centralized config dictionary.
    Keep minimal for now (paths are driven by CLI), but include
    detection and preprocessing parameters we want consistent
    """
    config = {
        "detector": {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": (60, 60)
        },
        "preprocess": {
            "ensure_rgb": True,
            "max_long_edge": 1600 #resize cap to control CPU/memory usage
        },
        "crop": {
            "margin_pct": 0.20 # simple margin around detected face
        },
        "defaults": {
            "schema_version": "schema_v1"
        }
    }
    return config

# Discovery & I/O

from pathlib import Path

def discover_images(root_dir: str) -> List[ImageRecord]:
    root = Path(root_dir)
    exts = {".jpg", ".jpeg", ".png"}

    # Debug: confirm the folder is resolvable
    print(f"[DEBUG] pathlib root exists: {root.exists()} is_dir: {root.is_dir()}")
    all_paths = list(root.rglob("*"))
    print(f"[DEBUG] pathlib rglob count: {len(all_paths)}")

    image_paths = [p for p in all_paths if p.is_file() and p.suffix.lower() in exts]
    print(f"[DEBUG] image_paths count: {len(image_paths)}")
    for p in image_paths[:5]:
        print(f"[DEBUG] sample: {p}")

    results = [ImageRecord(employee_id=p.stem, image_path=str(p)) for p in image_paths]
    return sorted(results, key=lambda r: (r.employee_id.lower(), r.image_path.lower()))



def validate_and_load(image_path: str) -> Any:
    """
    Safely load an image from disk and ensure:
    - Image Readability (returns None on failure)
    - Format conversion (grayscale, RGBA, BGR -> RGB)
    - Alpha compositing (transparent PNGs blended on white background)
    - Data type normalization (uint8)
    - Memory layout (C-contiguous array for dlib compatibililty)
    - Dimension validation (minimum size 100x100)

    Returns a numpy ndarray (uint8, C-contiguous) or None on failure.

    Uses OpenCV for loading to ensure dlib compatibility.
    """
    try:
        # Load with OpenCV (handles RGBA properly with IMREAD_COLOR flag)
        # IMREAD_UNCHANGED preserves alpha, IMREAD_COLOR converts to BGR
        img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img_bgr is None:
            print(f"[ERROR] Failed to read image with OpenCV: {image_path}")
            return None

        # Handle different channel configurations
        if img_bgr.ndim == 2:
            # Grayscale -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.ndim == 3:
            if img_bgr.shape[2] == 4:
                # RGBA/BGRA -> RGB (composite alpha on white background)
                # Extract alpha channel
                bgr = img_bgr[:, :, :3]
                alpha = img_bgr[:, :, 3:4] / 255.0
                # Composite on white background
                white_bg = np.ones_like(bgr) * 255
                img_bgr = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif img_bgr.shape[2] == 3:
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                print(f"[ERROR] Unsupported channel count: {img_bgr.shape[2]}")
                return None
        else:
            print(f"[ERROR] Unsupported image dimensions: {img_bgr.ndim}")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to read image: {image_path} ({e})")
        return None

    # Ensure uint8 and contiguity
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    img_rgb = np.ascontiguousarray(img_rgb)

    h, w = img_rgb.shape[:2]
    if h < 100 or w < 100:
        print(f"[ERROR] Image too small (<100x100): {image_path} ({w}x{h})")
        return None

    return img_rgb



def preprocess_image(img: Any, max_long_edge: int = 1600) -> Any:
    """
    Downsize images to control memory and CPU usage during face detection.

    Only resizes if longer dimension exceeds max_long_edge.
    Preserves aspect ratio using INTER_AREA interpolation (highest quality for downsampling).

    Args:
    - img: RGB image (from validate_and_load)
    - max_long_edge: max pixel size for longer dimension (default: 1600)

    Returns:
        Resized RBG image (original if already <= max_long_edge).
        
    Skip advanced normalization for now.
    """
    
    h,w = img.shape[:2]
    long_edge = max(h,w)
    if long_edge <= max_long_edge:
        return img

    scale = max_long_edge / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized


# Detection and Cropping

def detect_faces(img: Any, cascade_path: str, detector_params: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Use OpenCV Haar cascade to detect faces in image.
    Return list of bounding boxes (x, y, w, h).
    CLI Policy: proceed only if exactly one face; otherwise log and skip.
    """
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(cascade_path)
    
    if classifier.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    boxes = classifier.detectMultiScale(
        gray,
        scaleFactor=detector_params.get("scale_factor", 1.1),
        minNeighbors=detector_params.get("min_neighbors", 5),
        minSize=detector_params.get("min_size", (60, 60)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return [ (int(x), int(y), int(w), int(h)) for (x,y,w,h) in boxes]


def crop_with_margin(img: Any, box: Tuple[int, int, int, int],margin_pct: float = 0.20) -> Any:
    """
    Simple crop around the detected face with a configurable margin.
    Clamp to image bounds. No alignment for MVP.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = box
    mx = int(round(bw * margin_pct))
    my = int(round(bh * margin_pct))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + bw + mx)
    y1 = min(h, y + bh + my)
    roi = img[y0:y1, x0:x1]
    return roi

# Embedding computation

def compute_embedding(face_roi: np.ndarray) -> Optional[List[float]]:
    # 1) Normalize channels (handle grayscale / RGBA)
    if face_roi is None:
        print("[ERROR] ROI is None")
        return None
    if face_roi.ndim == 2:
        # Grayscale -> RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    elif face_roi.ndim == 3:
        if face_roi.shape[2] == 4:
            # RGBA -> RGB (drop alpha channel)
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGBA2RGB)
        elif face_roi.shape[2] == 3:
            # Already RGB from face_recognition.load_image_file() - no conversion needed
            # The image is loaded via PIL (RGB format), not OpenCV (BGR format)
            pass
        else:
            print(f"[ERROR] Unsupported channel count: {face_roi.shape}")
            return None
    else:
        print(f"[ERROR] Unsupported ROI shape: {face_roi.shape}")
        return None

    # 2) Normalize dtype to uint8 (handles 16-bit)
    if face_roi.dtype == np.uint16:
        face_roi = (face_roi / 257.0).astype(np.uint8)
    elif face_roi.dtype != np.uint8:
        face_roi = face_roi.astype(np.uint8)

    # 3) Ensure C-contiguous for dlib
    face_roi = np.ascontiguousarray(face_roi)

    # 4) Encode
    encodings = face_recognition.face_encodings(face_roi)
    if not encodings:
        return None
    return encodings[0].tolist()

# Record building and serialization

def build_record(employee_id: str, image_path: str, encoding: List[float], box: Tuple[int, int, int, int]) -> FaceRecord:
    """
    Package a single face record for downstream aggregation/serialization.
    """
    return FaceRecord(
        label=employee_id,
        encoding=encoding,
        image_path=image_path,
        box=box
    )

def serialize(records: List[FaceRecord], output_pkl_path: str, schema_version: str = "schema_v1") -> None:
    """
    Persist EncodingsDB to output_pkl_path.
    DB layout (MVP):
       - encodings: List[List[float]] (List of 128-d float lists)
       - labels: List[str] (employee IDs)
       - meta: List[Dict[str, Any]] (image_path, box, etc.) with minimal fields
       - version: schema_version
    """
    encs = [r.encoding for r in records]
    labels = [r.label for r in records]
    meta = [{"image_path": r.image_path, "box": r.box} for r in records]
    db = EncodingsDB(encodings=encs, labels=labels, meta=meta, version=schema_version)

    out_dir = os.path.dirname(output_pkl_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

# Incremental merge (skeleton)

def merge_incremental(existing: EncodingsDB, new_Records: List[FaceRecord]) -> EncodingsDB:
    """
    Barebones skeletion for future incremental merge.
    MVP behavior (not used by default):
       - Strategy TBD: considering deduplication by file content hash or image_path
       - For now, could simply append new unique paths, or rebuild from scratch each time.
    Simple path-based dedup:
        - Build a set of existing image_paths from meta
        - Append only new unique paths
    """
    existing_paths = {m.get("image_path") for m in existing.meta}
    merged_records: List[FaceRecord] = []

    # Reconstruct existing records from DB for uniformity
    for enc, label, m in zip(existing.encodings, existing.labels, existing.meta):
        merged_records.append(FaceRecord(label=label, encoding=enc, image_path=m["image_path"], box=tuple(m["box"])))
    
    for r in new_records:
        if r.image_path not in existing_paths:
            merged_records.append(r)
        
    encs = [r.encoding for r in merged_records]
    labels = [r.label for r in merged_records]
    meta = [{"image_path": r.image_path, "box": r.box} for r in merged_records]
    return EncodingsDB(encodings=encs, labels=labels, meta=meta, version=existing.version)

# CLI wiring


def parse_args() -> argparse.Namespace:
    """
    Define CLI parameters for repeatable runs.
    """
    parser = argparse.ArgumentParser(
        description="Build face embeddings from a Dropbox folder and serialize to known_faces.pkl"
    )
    parser.add_argument("--root", required=True, help="Root folder containing employee images (PNG/JPG)")
    parser.add_argument("--output", default="data/known_faces.pkl", help="Path to output pickle file")
    parser.add_argument("--cascade", required=True, help="Path to Haar Cascade XML (e.g., haarcascade_frontalface_default.xml)")
    parser.add_argument("--margin", type=float, default=0.20, help="Crop margin percentage around detected face")
    parser.add_argument("--max-long-edge", type=int, default=1600, help="Resize cap for the longer image edge")
    parser.add_argument("--rebuild", action="store_true", help="Ignore any existing DB and rebuild from scratch")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--incremental", action="store_true", help="(Future) Merge new images into existing DB")
    return parser.parse_args()


def encode_single_image(args_tuple):
    """
    Worker function for multiprocessing. Processes one image end-to-end.

    Args:
        args_tuple: (ImageRecord, config_dict) — single tuple for pool.map()

    Returns:
        FaceRecord on success, or a string skip-reason on failure.
    """
    image_record, config = args_tuple
    cascade_path = config["cascade_path"]
    max_long_edge = config["max_long_edge"]
    detector_params = config["detector"]

    # 1. Validate and load
    img = validate_and_load(image_record.image_path)
    if img is None:
        return "invalid"

    # 2. Preprocess
    img_prep = preprocess_image(img, max_long_edge=max_long_edge)

    # 3. Detect faces — require exactly 1
    boxes = detect_faces(img_prep, cascade_path, detector_params)
    if len(boxes) != 1:
        return "face_count"

    # 4. Convert Haar (x, y, w, h) -> face_recognition (top, right, bottom, left)
    x, y, w, h = boxes[0]
    fr_box = (y, x + w, y + h, x)

    # Pre-encode normalization
    if not (img_prep.ndim == 3 and img_prep.shape[2] == 3 and img_prep.dtype == np.uint8):
        if img_prep.ndim == 2:
            img_prep = cv2.cvtColor(img_prep, cv2.COLOR_GRAY2RGB)
        elif img_prep.ndim == 3 and img_prep.shape[2] == 4:
            img_prep = cv2.cvtColor(img_prep, cv2.COLOR_RGBA2RGB)
        img_prep = img_prep.astype(np.uint8)
        img_prep = np.ascontiguousarray(img_prep)

    # 5. Primary encoding path: known_face_locations
    try:
        encs = face_recognition.face_encodings(img_prep, known_face_locations=[fr_box])
    except Exception:
        encs = []

    # 6. Fallback: manual crop
    if not encs:
        top, right, bottom, left = fr_box
        H, W = img_prep.shape[:2]
        top    = max(0, min(H - 1, top))
        bottom = max(0, min(H,     bottom))
        left   = max(0, min(W - 1, left))
        right  = max(0, min(W,     right))

        face_roi = img_prep[top:bottom, left:right]
        if face_roi is None or face_roi.size == 0:
            return "encoding_fail"

        if face_roi.ndim == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        elif face_roi.ndim == 3 and face_roi.shape[2] == 4:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGBA2RGB)
        face_roi = face_roi.astype(np.uint8)
        face_roi = np.ascontiguousarray(face_roi)

        try:
            encs = face_recognition.face_encodings(face_roi)
        except Exception:
            encs = []

    # 7. Final check
    if not encs:
        return "encoding_fail"

    encoding = encs[0].tolist()
    return build_record(image_record.employee_id, image_record.image_path, encoding, boxes[0])


def cli_main() -> None:
    """
    Orchestrates the end-to-end run:
      1) Load config
      2) Discover images
      3) For each image:
           - validate & load
           - preprocess
           - detect faces
           - enforce exactly-one-face policy (else log/skip)
           - crop with margin
           - compute embedding
           - collect record
      4) (Optional future) merge incremental if requested and existing DB is present
      5) Serialize database
      6) Report summary
    """
    args = parse_args()
    print("Parsed arguments:", args)
    config = load_config()

    # Normalize CLI Params
    root_dir = args.root.strip().strip('"').strip("'")
    output_pkl = args.output
    cascade_path = args.cascade
    margin_pct = args.margin
    max_long_edge = args.max_long_edge
    rebuild = args.rebuild
    verbose = args.verbose
    incremental = args.incremental

    
    if verbose:
        print(f"[INFO] Root: {root_dir}")
        print(f"[INFO] Output: {output_pkl}")
        print(f"[INFO] Cascade: {cascade_path}")
        print(f"[INFO] Margin: {margin_pct}")
        print(f"[INFO] Max long edge: {max_long_edge}")
        print(f"[INFO] Rebuild: {rebuild} | Incremental: {incremental}")

    
    # Pathlib sanity check (new)
    from pathlib import Path
    print(f"[DEBUG] normalized root repr: {repr(root_dir)}")
    print(f"[DEBUG] Path(root).exists: {Path(root_dir).exists()}  is_dir: {Path(root_dir).is_dir()}")

    images = discover_images(root_dir)
    print(f"[DEBUG] cli_main got {(len(images))} image(s)")
    if images[:5]:
        for r in images[:5]:
            print(f"[DEBUG] sample image: {r.image_path}")

    if verbose:
        print(f"[INFO] Discovered {len(images)} image(s).")

    
    # Build worker config (includes cascade_path so workers can load their own classifier)
    worker_config = {
        "cascade_path": cascade_path,
        "max_long_edge": max_long_edge,
        "detector": config["detector"],
    }
    work_items = [(rec, worker_config) for rec in images]

    from multiprocessing import Pool, cpu_count
    num_workers = max(1, cpu_count() - 1)
    total = len(work_items)
    print(f"Encoding {total} images using {num_workers} workers...")

    new_records: List[FaceRecord] = []
    skipped_invalid = 0
    skipped_face_count = 0
    skipped_encoding_fail = 0

    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(encode_single_image, work_items), 1):
            print(f"\rProcessed {i}/{total}", end="", flush=True)
            if isinstance(result, FaceRecord):
                new_records.append(result)
            elif result == "invalid":
                skipped_invalid += 1
            elif result == "face_count":
                skipped_face_count += 1
            elif result == "encoding_fail":
                skipped_encoding_fail += 1

    print()  # newline after progress


    # If incremental merge requested and output exists (and not rebuilding), merge
    if incremental and not rebuild and os.path.isfile(output_pkl):
        if verbose:
            print(f"[INFO] Loading existing DB for incremental merge: {output_pkl}")
        with open(output_pkl, "rb") as f:
            existing_db: EncodingsDB = pickle.load(f)
        merged_db = merge_incremental(existing_db, new_records)
        # Reconstruct FaceRecord list to reuse serialize()
        merged_records: List[FaceRecord] = [
            FaceRecord(label=label, encoding=enc, image_path=m["image_path"], box=tuple(m["box"]))
            for enc, label, m in zip(merged_db.encodings, merged_db.labels, merged_db.meta)
        ]
        serialize(merged_records, output_pkl, schema_version=merged_db.version)
    else:
        serialize(new_records, output_pkl, schema_version=config["defaults"]["schema_version"])

    # Summary
    print(f"[SUMMARY] Total images: {len(images)}")
    print(f"[SUMMARY] Encoded: {len(new_records)}")
    print(f"[SUMMARY] Skipped invalid image: {skipped_invalid}")
    print(f"[SUMMARY] Skipped face count != 1: {skipped_face_count}")
    print(f"[SUMMARY] Skipped encoding failures: {skipped_encoding_fail}")
    print(f"[DONE] Wrote: {output_pkl}")


if __name__ == "__main__":
    try:
        cli_main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

