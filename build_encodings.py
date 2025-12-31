# build_encodings.py

"""
Purpose: Precompute and package employee face embeddings into a single, fast-to-load file for runtime recognition.
Value: Speeds up recognition, isolates data prep from inference, improves reliability, and sets you up to scale cleanly.
Lifecycle: Collect photos → build encodings (.pkl) → recognize at runtime → repeat as your dataset improves.
"""
from __future__ import annotations
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
    version: str = "schema_v1""


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

def discover_images(root_dir: str) -> List[ImageRecord]:
    """
    Recursively discover .jpg/.jpeg/.png images under root_dir.
    Derive employee_id from filename or parent folder (final decision TBD).
    """
    exts = {".jpg", ".jpeg", ".png"}
    results: List[ImageRecord] = []

    for dirpaths, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                path = os.path.join(dirpath, fn)
                employee_id = os.path.splitext(os.path.basename(fn))[0]
                results.append(ImageRecord(employee_id=employee_id, image_path=path))

    return sorted(results, key=lambda r: (r.employee_id.lower(), r.image_path.lower()))

def validate_and_load(image_path:str) -> Any:
    """
    Safeky load an image from disk and ensure:
       - it's readable
       - 8-bit RGB (convert if needed)
       - Reasonable dimensions (e.g, > 100x100)
    Return a numpy ndarray or raise/return None on failure.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None
    
    # Convert BGR -> RGB
    img_rbg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ensure Uint8
    if img_rbg.dtype != np.uint8:
        try:
            img_rgb = img_rbg.astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Failed to convert image to uint8: {image_path} ({e})")
            return None
    
    h,w = img_rgb.shape[:2]
    if h < 100 or w < 100:
        print(f"[ERROR] Image too small (<100x100): {image_path} ({w}x{h})")
        return None
    
    return img_rgb


def preprocess_image(img: Any, max_long_edge: int = 1600) -> Any:
    """
    Minimal preprocessing for MVP:
       - Ensure RGB(if not already)
       - Resize so the longer edge <= max_long_edge (preserving aspect ratio)
    Skip advanced normalization for now.
    """
    pass

# Detection and Cropping

def detect_faces(img: Any, cascade_path: str, detector_params: Dict[str, Any]) ->
List[Tuple[int, int, int, int]]:
    """
    Use OpenCV Haar cascade to detect faces in image.
    Return list of bounding boxes (x, y, w, h).
    CLI Policy: proceed only if exactly one face; otherwise log and skip.
    """
    pass

def crop_with_margin(img: Any, box: Tuple[int, int, int, int],margin_pct: float = 0.20) -> Any:
    """
    Simple crop around the detected face with a configurable margin.
    Clamp to image bounds. No alignment for MVP.
    """
    pass

# Embeddings

def build_record(employee_id: str, image_path: str, encoding: List[float], box: Tuple[int, int, int, int]) -> FaceRecord:
    """
    Package a single face rtecord for downstream aggregation/serialization.
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
    pass

# Incremental merge (skeleton)

def merge_incremental(existing: EncodingsDB, new_Records: List[FaceRecord]) -> EncodingsDB:
    """
    Barebones skeletion for future incremental merge.
    MVP behavior (not used by default):
       - Strategy TBD: considering deduplication by file content hash or image_path
       - For now, could simply append new unique paths, or rebuild from scratch each time.
    """
    pass

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
    # Future: --incremental flag; for now, we’ll parse but not use it
    parser.add_argument("--incremental", action="store_true", help="(Future) Merge new images into existing DB")
    return parser.parse_args()


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
    config = load_config()

    # Placeholder flow with clear checkpoints; implement step-by-step later.
    # root_dir = args.root
    # output_pkl = args.output
    # cascade_path = args.cascade
    # margin_pct = args.margin
    # max_long_edge = args.max_long_edge
    # rebuild = args.rebuild
    # verbose = args.verbose
    # incremental = args.incremental

    # images = discover_images(root_dir)
    # for rec in images:
    #     img = validate_and_load(rec.image_path)
    #     img_prep = preprocess_image(img, max_long_edge=max_long_edge)
    #     boxes = detect_faces(img_prep, cascade_path, config["detector"])
    #     if len(boxes) != 1:
    #         # log and continue
    #         continue
    #     face_roi = crop_with_margin(img_prep, boxes[0], margin_pct=margin_pct)
    #     encoding = compute_embedding(face_roi)
    #     if encoding is None:
    #         # log and continue
    #         continue
    #     # collect record(s)
    #     _ = build_record(rec.employee_id, rec.image_path, encoding, boxes[0])

    # # Optionally merge with existing DB (future)
    # # if incremental and not rebuild:
    # #     existing_db = ...  # load
    # #     merged = merge_incremental(existing_db, new_records)
    # #     serialize(merged, output_pkl)
    # # else:
    # #     serialize(new_records, output_pkl)

    pass


if __name__ == "__main__":
    cli_main()
