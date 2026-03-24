#!/usr/bin/env python3
"""
Multi-camera RTSP face recognition with SQLite logging.

Runs one process per camera, each with its own detector, embedder, and tracker.
Detections are logged to a shared SQLite database (WAL mode for safe concurrent writes).
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from multiprocessing import Process, Event
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from detector_factory import create_detector
from embedding_factory import create_embedder
from tracker import SimpleTracker, Detection
from build_encodings import EncodingsDB          # needed for pickle deserialization
from recognize import load_database, process_frame


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    name: str
    rtsp_url: str
    location: str

def load_camera_config(config_path: str) -> List[CameraConfig]:
    """Load camera list from a JSON config file."""
    with open(config_path, "r") as f:
        data = json.load(f)

    if "cameras" not in data or not isinstance(data["cameras"], list):
        raise ValueError(f"Config must contain a 'cameras' array: {config_path}")
    
    cameras = []
    for i, cam in enumerate(data["cameras"]):
        for field in ("name", "rtsp_url", "location"):
            if field not in cam:
                raise ValueError(f"Camera {i} missing required field '{field}'")
        cameras.append(CameraConfig(
            name=cam["name"],
            rtsp_url=cam["rtsp_url"],
            location=cam["location"]
        ))
    
    if not cameras: raise ValueError("Config contains no cameras")

    return cameras


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

def init_database(db_path: str) -> sqlite3.Connection:
    """Create detections table and indexes, enable WAL mode.

    Returns an open connection ready for use.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            camera_name TEXT   NOT NULL,
            location   TEXT    NOT NULL,
            identity   TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            distance   REAL    NOT NULL,
            bbox_x     INTEGER,
            bbox_y     INTEGER,
            bbox_w     INTEGER,
            bbox_h     INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_identity ON detections(identity)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_camera ON detections(camera_name)")
    conn.commit()

    return conn


def log_detection(conn: sqlite3.Connection,
                  camera_name: str,
                  location: str,
                  detection: Detection,
                  timestamp: str) -> None:
    """Insert a single detection row. Caller is responsible for committing."""
    x, y, w, h = detection.bbox
    conn.execute(
        "INSERT INTO detections "
        "(timestamp, camera_name, location, identity, confidence, distance, "
        " bbox_x, bbox_y, bbox_w, bbox_h) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, camera_name, location,
         detection.label, detection.confidence, detection.distance,
         x, y, w, h),
    )


def purge_old_detections(db_path: str, retention_days: int) -> int:
    """Delete detections older than *retention days*. Returns rows deleted."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "DELETE FROM detections WHERE timestamp < ?", (cutoff,)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Camera worker (one per RTSP stream)
# ---------------------------------------------------------------------------

def camera_worker(camera: CameraConfig,
                  shared_config: Dict[str, Any],
                  shutdown_event: Event) -> None:
    """Process a single RTSP stream: detect, track, recognize, and log.

    Each worker runs in its own process with its own detector, embedder,
    tracker, and SQLite connection.
    """
    # Children must ignore SIGINT — only the main process handles Ctrl+C
    # and coordinates shutdown via shutdown_event.  Without this, a
    # KeyboardInterrupt can land mid-inference or mid-SQLite-write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log = logging.getLogger(f"cam.{camera.name}")
    log.info("Worker starting")

    # --- Per-process setup (cannot share across processes) ---
    detector = create_detector(
        shared_config["detector_type"],
        cascade_path=shared_config["cascade_path"],
    )
    embedder = create_embedder(
        shared_config["embedder_type"],
        model_name=shared_config["model_name"],
        ctx_id=shared_config["ctx_id"],
    )
    known_encodings, labels = load_database(
        shared_config["database_path"],
        expected_embedder=shared_config["embedder_type"],
    )
    conn = init_database(shared_config["sqlite_path"])

    resize_width = shared_config["resize_width"]
    threshold = shared_config["threshold"]
    do_align = shared_config["do_align"]
    max_retries = shared_config["max_retries"]
    log_interval = shared_config["log_interval"]
    tracker_interval = shared_config["tracker_interval"]

    # FFMPEG: TCP transport + 10s timeout
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|timeout;10000000"
    )

    # --- Reconnection loop ---
    attempts = 0

    try:
        while attempts <= max_retries and not shutdown_event.is_set():
            if attempts == 0:
                log.info("Connecting to %s", camera.rtsp_url)
            else:
                log.info("Reconnection attempt %d/%d", attempts, max_retries)

            cap = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                cap.release()
                attempts += 1
                if attempts > max_retries:
                    break
                log.warning("Connection failed, retrying in 2s")
                time.sleep(2)
                continue

            log.info("Stream connected")
            attempts = 0

            tracker = SimpleTracker(reidentify_interval=tracker_interval)
            last_logged: Dict[str, float] = {}   # identity -> last log time
            last_commit = time.time()

            try:
                while not shutdown_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("Lost stream")
                        break

                    # Resize for performance
                    if resize_width and frame.shape[1] > resize_width:
                        scale = resize_width / frame.shape[1]
                        frame = cv2.resize(
                            frame, (resize_width, int(frame.shape[0] * scale))
                        )

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Fast detection every frame
                    current_boxes = tracker.detect_faces(frame_rgb, detector)

                    if tracker.should_reidentify(current_boxes):
                        detections = process_frame(
                            frame_rgb, known_encodings, labels,
                            detector, threshold, do_align, embedder,
                        )
                        tracker.update(
                            boxes=[d.bbox for d in detections],
                            labels=[d.label for d in detections],
                            confidences=[d.confidence for d in detections],
                            distances=[d.distance for d in detections],
                        )
                    else:
                        detections = tracker.get_cached_detections(current_boxes)

                    # Log detections with dedup
                    now = time.time()
                    ts = datetime.now(timezone.utc).isoformat()

                    for det in detections:
                        since = now - last_logged.get(det.label, 0)
                        if since >= log_interval:
                            log_detection(conn, camera.name, camera.location,
                                          det, ts)
                            last_logged[det.label] = now

                    # Batch commit every 5 seconds
                    if now - last_commit >= 5.0:
                        conn.commit()
                        last_commit = now

            finally:
                cap.release()

            # Stream lost — attempt reconnection
            if shutdown_event.is_set():
                break
            attempts += 1
            if attempts <= max_retries:
                log.info("Reconnecting in 2s")
                time.sleep(2)

    except Exception as exc:
        log.error("Unexpected worker error: %s", exc, exc_info=True)
        raise

    finally:
        # Flush remaining rows and close, even on unexpected exit
        conn.commit()
        conn.close()

    if attempts > max_retries:
        log.error("Failed to reconnect after %d attempts", max_retries)

    log.info("Worker stopped")


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

MAX_RESTARTS = 3          # per-camera restart limit before giving up


def run_multi(cameras: List[CameraConfig],
              shared_config: Dict[str, Any]) -> None:
    """Spawn one process per camera, monitor them, and handle shutdown."""
    log = logging.getLogger("multi")

    # Ensure database and table exist before any operations
    sqlite_path = shared_config["sqlite_path"]
    conn = init_database(sqlite_path)
    conn.close()

    # Purge old detections on startup
    retention_days = shared_config.get("retention_days", 0)
    if retention_days > 0:
        deleted = purge_old_detections(sqlite_path, retention_days)
        if deleted:
            log.info("Purged %d detections older than %d days",
                     deleted, retention_days)

    # Shared event — main process sets it on SIGINT/SIGTERM,
    # children check it each frame iteration.
    shutdown_event = Event()

    def _on_signal(signum, _frame):
        # No logging here — acquiring a lock inside a signal handler
        # can deadlock if the signal interrupts another logging call.
        shutdown_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Spawn one process per camera ---
    cam_by_name: Dict[str, CameraConfig] = {c.name: c for c in cameras}
    procs: Dict[str, Process] = {}
    restarts: Dict[str, int] = {c.name: 0 for c in cameras}

    log.info("Starting %d camera(s)", len(cameras))

    for cam in cameras:
        p = Process(
            target=camera_worker,
            args=(cam, shared_config, shutdown_event),
            name=f"cam-{cam.name}",
        )
        p.start()
        procs[cam.name] = p
        log.info("  [%s] pid %d", cam.name, p.pid)

    # --- Monitor loop ---
    # Sleep in short increments so Ctrl+C can interrupt — on macOS,
    # multiprocessing.Event.wait() can block signal delivery.
    while not shutdown_event.is_set():
        for _ in range(50):                # 50 × 0.1s = 5s between checks
            if shutdown_event.is_set():
                break
            time.sleep(0.1)
        if shutdown_event.is_set():
            break

        for name, proc in list(procs.items()):
            if proc.is_alive():
                continue

            code = proc.exitcode
            log.warning("[%s] exited unexpectedly (code %s)", name, code)

            if restarts[name] >= MAX_RESTARTS:
                log.error("[%s] exceeded %d restarts — giving up",
                          name, MAX_RESTARTS)
                continue

            restarts[name] += 1
            log.info("[%s] restarting (%d/%d)",
                     name, restarts[name], MAX_RESTARTS)

            cam = cam_by_name[name]
            p = Process(
                target=camera_worker,
                args=(cam, shared_config, shutdown_event),
                name=f"cam-{cam.name}",
            )
            p.start()
            procs[name] = p
            log.info("  [%s] pid %d", name, p.pid)

        # Check if any camera is still running or can be restarted
        any_alive = any(p.is_alive() for p in procs.values())
        any_restartable = any(
            not procs[n].is_alive() and restarts[n] < MAX_RESTARTS
            for n in procs
        )
        if not any_alive and not any_restartable:
            log.error("All cameras dead — nothing left to manage")
            break

    # --- Graceful shutdown ---
    log.info("Shutdown signal received")
    log.info("Stopping %d process(es)", len(procs))

    for name, proc in procs.items():
        proc.join(timeout=3)
        if proc.is_alive():
            log.warning("[%s] still blocked (likely in cap.read), terminating", name)
            proc.terminate()
            proc.join(timeout=2)

    log.info("All cameras stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-camera RTSP face recognition with SQLite logging."
    )
    p.add_argument("--config", required=True,
                   help="Path to cameras JSON config file")
    p.add_argument("--database", default="data/known_faces.pkl",
                   help="Face encodings database (.pkl)")
    p.add_argument("--sqlite", default="data/detections.db",
                   help="SQLite database path for logging detections")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Distance threshold for face matching")
    p.add_argument("--detector", choices=["haar", "retinaface"], default="haar",
                   help="Face detector backend")
    p.add_argument("--embedder", choices=["dlib", "arcface"], default="dlib",
                   help="Face embedder backend")
    p.add_argument("--align", action="store_true",
                   help="Enable landmark-based face alignment")
    p.add_argument("--model", default="buffalo_l",
                   help="InsightFace model pack name")
    p.add_argument("--gpu", type=int, default=-1,
                   help="GPU device ID (-1 for CPU)")
    p.add_argument("--cascade",
                   default="data/haarcascade_frontalface_default.xml",
                   help="Haar cascade XML path")
    p.add_argument("--resize-width", type=int, default=640,
                   help="Resize frames to this width (0 to disable)")
    p.add_argument("--tracker-interval", type=int, default=30,
                   help="Frames between re-identification")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Max RTSP reconnection attempts per camera")
    p.add_argument("--retention-days", type=int, default=30,
                   help="Auto-purge detections older than N days (0 to disable)")
    p.add_argument("--log-interval", type=float, default=60.0,
                   help="Min seconds between logging the same identity")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("main")

    # --- Validate inputs ---
    if not os.path.isfile(args.config):
        log.error("Config file not found: %s", args.config)
        sys.exit(1)
    if not os.path.isfile(args.database):
        log.error("Encodings database not found: %s", args.database)
        sys.exit(1)

    # Ensure SQLite parent directory exists
    sqlite_dir = os.path.dirname(args.sqlite)
    if sqlite_dir and not os.path.isdir(sqlite_dir):
        os.makedirs(sqlite_dir, exist_ok=True)

    cameras = load_camera_config(args.config)
    log.info("Loaded %d camera(s) from %s", len(cameras), args.config)

    shared_config = {
        "database_path":  args.database,
        "sqlite_path":    args.sqlite,
        "threshold":      args.threshold,
        "detector_type":  args.detector,
        "embedder_type":  args.embedder,
        "do_align":       args.align,
        "model_name":     args.model,
        "ctx_id":         args.gpu,
        "cascade_path":   args.cascade,
        "resize_width":   args.resize_width,
        "tracker_interval": args.tracker_interval,
        "max_retries":    args.max_retries,
        "retention_days": args.retention_days,
        "log_interval":   args.log_interval,
    }

    run_multi(cameras, shared_config)


if __name__ == "__main__":
    main()