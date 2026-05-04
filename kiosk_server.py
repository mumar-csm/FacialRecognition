#!/usr/bin/env python3
"""
Phase K2 — Kiosk Recognition Server

FastAPI-based kiosk for employee clock-in/out via facial recognition.
Loads all ML models once at startup, serves a browser-based frontend,
and exposes REST endpoints for recognition and attendance.

Usage:
    python kiosk_server.py --database data/known_faces_arcface.pkl
    python kiosk_server.py --database data/known_faces_arcface.pkl --threshold 0.9 --cooldown 120
"""

import argparse
import base64
import csv
import io
import os
import pickle
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Reuse existing pipeline components ──
from anti_spoof_factory import create_anti_spoof
from build_encodings import EncodingsDB
from detector_factory import FaceDetector, align_face, create_detector
from embedding_factory import create_embedder
from euclideanDist import euclidean_distance
from liveness import LivenessManager, SessionState
from recognize import find_best_match, load_database


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_local(ts: str, tz: ZoneInfo) -> str:
    """Convert a UTC ISO timestamp string to the given local timezone (no tz suffix)."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return ts


# ═══════════════════════════════════════════════════════════════════════
# SQLite Database Layer
# ═══════════════════════════════════════════════════════════════════════

def init_kiosk_db(db_path: str) -> sqlite3.Connection:
    """Create kiosk tables and indexes, enable WAL mode. Returns open connection."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            enrolled_at TEXT NOT NULL,
            photo_path  TEXT,
            is_active   INTEGER NOT NULL DEFAULT 1,
            store_id    TEXT NOT NULL DEFAULT 'store-01'
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            employee_id TEXT NOT NULL,
            distance    REAL NOT NULL,
            is_clock_in INTEGER NOT NULL,
            camera_id   TEXT NOT NULL DEFAULT 'kiosk-01',
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        );

        CREATE TABLE IF NOT EXISTS spoof_attempts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            camera_id   TEXT NOT NULL DEFAULT 'kiosk-01',
            spoof_score REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp);
        CREATE INDEX IF NOT EXISTS idx_attendance_employee  ON attendance(employee_id);
        CREATE INDEX IF NOT EXISTS idx_spoof_timestamp      ON spoof_attempts(timestamp);
    """)

    # Migration: add store_id to existing databases that predate this column
    try:
        conn.execute("ALTER TABLE employees ADD COLUMN store_id TEXT NOT NULL DEFAULT 'store-01'")
    except sqlite3.OperationalError:
        pass  # column already exists

    conn.commit()
    return conn


def log_attendance(conn: sqlite3.Connection, employee_id: str,
                   distance: float, is_clock_in: bool, camera_id: str = "kiosk-01") -> int:
    """Insert an attendance record. Returns the inserted row ID."""
    ts = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO attendance (timestamp, employee_id, distance, is_clock_in, camera_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (ts, employee_id, distance, int(is_clock_in), camera_id),
    )
    conn.commit()
    return cursor.lastrowid


def log_spoof_attempt(conn: sqlite3.Connection, camera_id: str, spoof_score: float) -> int:
    """Insert a spoof attempt record. Returns the inserted row ID."""
    ts = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO spoof_attempts (timestamp, camera_id, spoof_score) VALUES (?, ?, ?)",
        (ts, camera_id, spoof_score),
    )
    conn.commit()
    return cursor.lastrowid


def get_last_attendance(conn: sqlite3.Connection, employee_id: str,
                        today_date: str) -> Optional[Dict]:
    """Get the most recent attendance record for an employee today.

    Args:
        today_date: "YYYY-MM-DD" — used for LIKE matching on timestamp.
    Returns:
        dict with is_clock_in field, or None if no records today.
    """
    row = conn.execute(
        "SELECT id, timestamp, employee_id, distance, is_clock_in, camera_id "
        "FROM attendance "
        "WHERE employee_id = ? AND timestamp LIKE ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (employee_id, f"{today_date}%"),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "timestamp": row[1], "employee_id": row[2],
        "distance": row[3], "is_clock_in": bool(row[4]), "camera_id": row[5],
    }


def get_attendance_by_date(conn: sqlite3.Connection, date: str) -> List[Dict]:
    """Get all attendance records for a given date (YYYY-MM-DD)."""
    rows = conn.execute(
        "SELECT a.id, a.timestamp, a.employee_id, a.distance, a.is_clock_in, a.camera_id "
        "FROM attendance a "
        "WHERE a.timestamp LIKE ? "
        "ORDER BY a.timestamp",
        (f"{date}%",),
    ).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "employee_id": r[2],
            "distance": r[3], "is_clock_in": bool(r[4]), "camera_id": r[5],
        }
        for r in rows
    ]


def get_report_records(
    conn: sqlite3.Connection,
    start_bound: str,
    end_bound: str,
    employee: Optional[str] = None,
) -> List[Dict]:
    """Attendance records for a date range, joined with employee names."""
    sql = (
        "SELECT a.id, a.timestamp, a.employee_id, "
        "COALESCE(e.name, a.employee_id) AS employee_name, "
        "a.distance, a.is_clock_in, a.camera_id "
        "FROM attendance a "
        "LEFT JOIN employees e ON e.id = a.employee_id "
        "WHERE a.timestamp >= ? AND a.timestamp <= ? "
    )
    params: list = [start_bound, end_bound]
    if employee:
        sql += "AND a.employee_id = ? "
        params.append(employee)
    sql += "ORDER BY a.timestamp ASC"
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r[0], "timestamp": r[1], "employee_id": r[2],
            "employee_name": r[3], "distance": r[4],
            "is_clock_in": bool(r[5]), "camera_id": r[6],
        }
        for r in rows
    ]


def get_spoof_count_for_range(
    conn: sqlite3.Connection, start_bound: str, end_bound: str
) -> int:
    """Count spoof attempts in a date range."""
    row = conn.execute(
        "SELECT COUNT(*) FROM spoof_attempts WHERE timestamp >= ? AND timestamp <= ?",
        (start_bound, end_bound),
    ).fetchone()
    return row[0] if row else 0


def deactivate_employee(conn: sqlite3.Connection, employee_id: str) -> bool:
    """Soft-delete employee (is_active=0). Returns True if row was found and updated."""
    cursor = conn.execute(
        "UPDATE employees SET is_active = 0 WHERE id = ? AND is_active = 1",
        (employee_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_active_employees(conn: sqlite3.Connection, store_id: str) -> List[Dict]:
    """Return all active enrolled employees for a given store."""
    rows = conn.execute(
        "SELECT id, name, enrolled_at, photo_path FROM employees "
        "WHERE is_active = 1 AND store_id = ? ORDER BY name",
        (store_id,),
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "enrolled_at": r[2], "photo_path": r[3]}
        for r in rows
    ]


def purge_old_records(conn: sqlite3.Connection,
                      attendance_days: int = 365,
                      spoof_days: int = 90) -> Tuple[int, int]:
    """Delete records older than retention window. Returns (attendance_deleted, spoof_deleted)."""
    att_cutoff = (datetime.now(timezone.utc) - timedelta(days=attendance_days)).isoformat()
    spoof_cutoff = (datetime.now(timezone.utc) - timedelta(days=spoof_days)).isoformat()

    c1 = conn.execute("DELETE FROM attendance WHERE timestamp < ?", (att_cutoff,))
    c2 = conn.execute("DELETE FROM spoof_attempts WHERE timestamp < ?", (spoof_cutoff,))
    conn.commit()
    return c1.rowcount, c2.rowcount


# ═══════════════════════════════════════════════════════════════════════
# Enrollment persistence
# ═══════════════════════════════════════════════════════════════════════

def remove_encoding_from_pkl(db_path: str, label: str) -> None:
    """Remove all entries for a label from the pkl file (atomic write)."""
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    # Filter all three parallel lists together
    filtered = [
        (enc, lbl, meta)
        for enc, lbl, meta in zip(db.encodings, db.labels, db.meta)
        if lbl != label
    ]
    if filtered:
        db.encodings, db.labels, db.meta = map(list, zip(*filtered))
    else:
        db.encodings, db.labels, db.meta = [], [], []
    tmp_path = db_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp_path, db_path)


def save_encoding_to_pkl(db_path: str, embedding: np.ndarray, label: str) -> None:
    """Append one encoding to the pkl file on disk using atomic write."""
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    db.encodings.append(embedding.tolist())
    db.labels.append(label)
    db.meta.append({"source": "kiosk_enrollment", "label": label})
    tmp_path = db_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp_path, db_path)


# ═══════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Kiosk Recognition Server (K2)")
    p.add_argument("--database", default="data/known_faces_arcface.pkl",
                   help="Path to face encodings pkl file")
    p.add_argument("--sqlite", default="data/kiosk.db",
                   help="Path to SQLite database")
    p.add_argument("--detector", choices=["haar", "retinaface"], default="retinaface")
    p.add_argument("--embedder", choices=["dlib", "arcface"], default="arcface")
    p.add_argument("--anti-spoof", choices=["none", "minifas"], default="minifas",
                   dest="anti_spoof")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Distance threshold for face match")
    p.add_argument("--cooldown", type=int, default=120,
                   help="Cooldown between duplicate clock events (seconds)")
    p.add_argument("--consensus", type=int, default=3,
                   help="Consecutive frames required to confirm identity before liveness challenge (default: 3)")
    p.add_argument("--spoof-threshold", type=float, default=0.55,
                   dest="spoof_threshold",
                   help="Anti-spoof probability threshold (0-1, higher=stricter, default: 0.55)")
    p.add_argument("--challenge-timeout", type=float, default=8.0,
                   dest="challenge_timeout",
                   help="Seconds allowed for liveness challenge (default: 8.0)")
    p.add_argument("--camera-id", default="kiosk-01",
                   help="Identifier for this kiosk instance")
    p.add_argument("--retention-days", type=int, default=365,
                   help="Attendance record retention (days)")
    p.add_argument("--spoof-retention-days", type=int, default=90,
                   help="Spoof attempt record retention (days)")
    p.add_argument("--store-id", default="store-01",
                   dest="store_id",
                   help="Identifier for this store location (e.g. 'downtown-01')")
    p.add_argument("--enrollment-pin", default=None,
                   dest="enrollment_pin",
                   help="Manager PIN required to enroll/delete employees (optional)")
    p.add_argument("--timezone", default="UTC",
                   help="Local timezone for report timestamps (e.g. 'America/New_York')")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Global config (populated by parse_args at module load)
# ═══════════════════════════════════════════════════════════════════════

args = parse_args()

STATIC_DIR = Path(__file__).parent / "static"


# ═══════════════════════════════════════════════════════════════════════
# FastAPI App + Lifespan
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and database once at startup."""
    print("[STARTUP] Loading models and database...")

    # Validate timezone early — fail fast before touching ML models
    try:
        local_tz = ZoneInfo(args.timezone)
    except Exception:
        print(f"[ERROR] Unknown timezone '{args.timezone}'. Use a tz database name like 'America/New_York'.")
        sys.exit(1)

    # ML pipeline
    app.state.detector = create_detector(args.detector)
    app.state.embedder = create_embedder(args.embedder)
    app.state.anti_spoof = create_anti_spoof(
        args.anti_spoof,
        **({"threshold": args.spoof_threshold} if args.anti_spoof != "none" else {}),
    )
    app.state.do_align = (args.detector == "retinaface")

    # Face encodings
    encodings, labels = load_database(args.database, expected_embedder=args.embedder)
    app.state.known_encodings = encodings
    app.state.known_labels = labels

    # SQLite
    app.state.db_conn = init_kiosk_db(args.sqlite)
    att_del, spoof_del = purge_old_records(
        app.state.db_conn, args.retention_days, args.spoof_retention_days
    )
    if att_del or spoof_del:
        print(f"[STARTUP] Purged {att_del} old attendance, {spoof_del} old spoof records")

    # In-memory cooldown tracker
    app.state.cooldown: Dict[str, float] = {}

    # Consensus tracker for identity confirmation (pre-challenge)
    app.state.pending: Dict[str, Dict] = {}

    # Consecutive spoof-positive frame counter (debounce borderline scores)
    app.state.spoof_streak: int = 0

    # Liveness challenge manager
    app.state.liveness = LivenessManager(challenge_timeout=args.challenge_timeout)

    # Config accessible to endpoints
    app.state.threshold = args.threshold
    app.state.cooldown_seconds = args.cooldown
    app.state.consensus_required = args.consensus
    app.state.camera_id = args.camera_id
    app.state.store_id = args.store_id
    app.state.enrollment_pin = args.enrollment_pin
    app.state.local_tz = local_tz

    pin_status = "protected" if args.enrollment_pin else "open"
    print(f"[STARTUP] Ready — store={args.store_id}, tz={args.timezone}, "
          f"enrollment={pin_status}, {len(labels)} employees, threshold={args.threshold}, "
          f"cooldown={args.cooldown}s, consensus={args.consensus} frames, "
          f"spoof_threshold={args.spoof_threshold}, challenge_timeout={args.challenge_timeout}s")
    yield

    # Shutdown
    app.state.db_conn.commit()
    app.state.db_conn.close()
    print("[SHUTDOWN] Database connection closed.")


app = FastAPI(title="Kiosk Recognition Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════

class EnrollRequest(BaseModel):
    image: str        # base64-encoded JPEG (with or without data URI prefix)
    first_name: str
    last_name: str
    pin: Optional[str] = None  # manager PIN if server is PIN-protected


class EnrollResponse(BaseModel):
    status: str  # enrolled, unauthorized, no_face, multiple_faces, spoof_detected, error
    message: str = ""
    employee_name: str = ""  # stored as firstname_lastname (or firstname_lastname_2 on collision)


class VerifyPinRequest(BaseModel):
    pin: Optional[str] = None


class RecognizeRequest(BaseModel):
    image: str  # base64-encoded JPEG (with or without data URI prefix)
    camera_id: Optional[str] = None


class RecognizeResponse(BaseModel):
    status: str  # recognized, verifying, liveness_challenge, no_face, multiple_faces, spoof_detected, unknown, cooldown
    identity: Optional[str] = None
    distance: Optional[float] = None
    is_clock_in: Optional[bool] = None
    consensus_progress: Optional[int] = None
    consensus_required: Optional[int] = None
    challenge_type: Optional[str] = None  # "blink" or "nod"
    challenge_instruction: Optional[str] = None
    challenge_time_remaining: Optional[float] = None
    message: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_index():
    """Serve the kiosk frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/enroll")
async def serve_enroll():
    """Serve the enrollment page."""
    return FileResponse(str(STATIC_DIR / "enroll.html"))


@app.post("/api/verify-pin")
async def verify_pin(request: Request, body: VerifyPinRequest):
    """Verify a manager PIN. Used by the enrollment page gate.

    If the server has no PIN configured, returns valid=true unconditionally
    (the enroll/delete endpoints also skip the check in that case)."""
    state = request.app.state
    if not state.enrollment_pin:
        return {"valid": True, "protected": False}
    return {"valid": body.pin == state.enrollment_pin, "protected": True}


@app.get("/manage")
async def serve_manage():
    """Serve the manage employees page."""
    return FileResponse(str(STATIC_DIR / "manage.html"))


@app.get("/report")
async def serve_report():
    """Serve the manager report page."""
    return FileResponse(str(STATIC_DIR / "report.html"))


@app.get("/api/health")
async def health(request: Request):
    """Server health and configuration summary."""
    state = request.app.state
    return {
        "status": "ok",
        "store_id": state.store_id,
        "timezone": str(state.local_tz),
        "enrollment_protected": bool(state.enrollment_pin),
        "detector": args.detector,
        "embedder": args.embedder,
        "anti_spoof": args.anti_spoof,
        "employee_count": len(state.known_labels),
        "unique_employees": len(set(state.known_labels)),
        "threshold": state.threshold,
        "cooldown_seconds": state.cooldown_seconds,
        "consensus_required": state.consensus_required,
    }


@app.get("/api/attendance")
async def get_attendance(request: Request, date: Optional[str] = None):
    """Get attendance records for a date (default: today)."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = get_attendance_by_date(request.app.state.db_conn, date)
    return {"date": date, "records": records, "count": len(records)}


@app.get("/api/report")
async def get_report(
    request: Request,
    start: Optional[str] = Query(default=None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="End date YYYY-MM-DD"),
    employee: Optional[str] = Query(default=None, description="Filter by employee_id"),
):
    """Attendance report for a date range, with summary stats."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if end is None:
        end = today
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")

    date_re = r"^\d{4}-\d{2}-\d{2}$"
    if not re.fullmatch(date_re, start) or not re.fullmatch(date_re, end):
        return JSONResponse({"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400)

    start_bound = start + "T00:00:00"
    end_bound = end + "T23:59:59"
    conn = request.app.state.db_conn

    records = get_report_records(conn, start_bound, end_bound, employee or None)
    spoof_count = get_spoof_count_for_range(conn, start_bound, end_bound)

    local_tz = request.app.state.local_tz
    for r in records:
        r["timestamp"] = _to_local(r["timestamp"], local_tz)

    return {
        "start": start,
        "end": end,
        "employee_filter": employee or None,
        "store_id": request.app.state.store_id,
        "timezone": str(local_tz),
        "records": records,
        "count": len(records),
        "summary": {
            "total_clock_ins": sum(1 for r in records if r["is_clock_in"]),
            "total_clock_outs": sum(1 for r in records if not r["is_clock_in"]),
            "unique_employees": len(set(r["employee_id"] for r in records)),
            "spoof_attempts_count": spoof_count,
        },
    }


@app.get("/api/report/csv")
async def get_report_csv(
    request: Request,
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    employee: Optional[str] = Query(default=None),
):
    """Download attendance report as CSV."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if end is None:
        end = today
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")

    date_re = r"^\d{4}-\d{2}-\d{2}$"
    if not re.fullmatch(date_re, start) or not re.fullmatch(date_re, end):
        return JSONResponse({"error": "Invalid date format."}, status_code=400)

    records = get_report_records(
        request.app.state.db_conn,
        start + "T00:00:00",
        end + "T23:59:59",
        employee or None,
    )

    local_tz = request.app.state.local_tz
    store_id = request.app.state.store_id
    for r in records:
        r["timestamp"] = _to_local(r["timestamp"], local_tz)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", f"Time ({local_tz})", "Employee ID", "Employee Name",
                     "Event Type", "Confidence", "Camera", "Store"])
    for r in records:
        ts = r["timestamp"]
        confidence = f"{max(0.0, (1.0 - r['distance']) * 100):.2f}%"
        writer.writerow([
            ts[:10], ts[11:19], r["employee_id"], r["employee_name"],
            "Clock In" if r["is_clock_in"] else "Clock Out",
            confidence, r["camera_id"], store_id,
        ])

    filename = f"attendance_{start}_{end}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/recognize")
async def recognize(request: Request, body: RecognizeRequest):
    """Core recognition pipeline — decode, detect, anti-spoof, embed, match, log."""
    state = request.app.state
    camera_id = body.camera_id or state.camera_id
    state.liveness.cleanup_stale()

    # ── Step 1: Decode base64 → BGR → RGB ──
    try:
        image_data = body.image
        # Strip data URI prefix if present
        if "," in image_data and image_data.index(",") < 100:
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return RecognizeResponse(status="error", message="Failed to decode image")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return RecognizeResponse(status="error", message=f"Image decode failed: {e}")

    # ── Step 2: Detect faces ──
    detections = state.detector.detect(frame_rgb)

    # ── Step 3: Reject 0 or >1 faces ──
    if len(detections) == 0:
        return RecognizeResponse(status="no_face", message="No face detected")
    if len(detections) > 1:
        return RecognizeResponse(
            status="multiple_faces",
            message="Multiple faces detected. Please ensure only one person is at the kiosk.",
        )

    (x, y, w, h), landmarks = detections[0]

    # ── Step 3.5: Fast-path during active liveness challenge ──
    # Identity is already confirmed; all we need are the landmarks for the blink
    # check. Skipping anti-spoof + align + embed + match cuts per-frame cost by
    # ~60-70%, which is what makes short blinks catchable (fewer missed samples).
    active_identity = next(iter(state.liveness.sessions), None)
    if active_identity is not None:
        # Use the session's most recent distance as a placeholder for logging
        prior_distances = state.liveness.sessions[active_identity].distances
        placeholder_dist = prior_distances[-1] if prior_distances else 0.0
        liveness_state, info = state.liveness.process_frame(
            active_identity, landmarks, frame_rgb, placeholder_dist
        )

        if liveness_state == SessionState.CHALLENGE_ACTIVE:
            return RecognizeResponse(
                status="liveness_challenge",
                identity=active_identity,
                distance=round(placeholder_dist, 4),
                challenge_type=info["challenge_type"],
                challenge_instruction="Please BLINK",
                challenge_time_remaining=info["time_remaining"],
                message="Please BLINK",
            )

        if liveness_state == SessionState.FAILED:
            log_spoof_attempt(state.db_conn, camera_id, 0.0)
            return RecognizeResponse(
                status="spoof_detected",
                message="Liveness challenge failed.",
            )

        # VERIFIED — proceed to clock-in/out
        avg_distance = info["avg_distance"]
        now = time.time()
        last_seen = state.cooldown.get(active_identity, 0)
        if now - last_seen < state.cooldown_seconds:
            remaining = int(state.cooldown_seconds - (now - last_seen))
            return RecognizeResponse(
                status="cooldown",
                identity=active_identity,
                distance=round(avg_distance, 4),
                message=f"Already recorded. Please wait {remaining}s.",
            )
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_record = get_last_attendance(state.db_conn, active_identity, today_str)
        is_clock_in = last_record is None or not last_record["is_clock_in"]
        log_attendance(state.db_conn, active_identity, avg_distance, is_clock_in, camera_id)
        state.cooldown[active_identity] = now
        action = "Clocked In" if is_clock_in else "Clocked Out"
        print(f"[ATTENDANCE] {active_identity} — {action} (avg_dist={avg_distance:.4f})")
        return RecognizeResponse(
            status="recognized",
            identity=active_identity,
            distance=round(avg_distance, 4),
            is_clock_in=is_clock_in,
            message=f"{action}: {active_identity}",
        )

    # ── Step 4: Anti-spoof check ──
    # Skip during an active liveness challenge — mid-blink frames score low on
    # MiniFAS (eyes partially closed), which would abort the challenge incorrectly.
    if state.anti_spoof is not None and not state.liveness.sessions:
        face_crop = frame_rgb[y:y+h, x:x+w]
        if face_crop.size > 0:
            is_real, spoof_score = state.anti_spoof.check(face_crop)
            print(f"[ANTI-SPOOF] is_real={is_real}, score={spoof_score:.4f}")
            if not is_real:
                # Debounce: a single borderline dip shouldn't kill consensus.
                # Require 2 consecutive spoof-positive frames before aborting —
                # real attacks score low consistently, real faces only flicker.
                state.spoof_streak += 1
                if state.spoof_streak >= 2:
                    state.pending.clear()
                    state.spoof_streak = 0
                    log_spoof_attempt(state.db_conn, camera_id, spoof_score)
                    return RecognizeResponse(
                        status="spoof_detected",
                        message="Liveness check failed.",
                    )
                # First dip — skip this frame, keep consensus intact
                return RecognizeResponse(
                    status="no_face",
                    message="Hold still.",
                )
            else:
                state.spoof_streak = 0

    # ── Step 5: Align face ──
    if state.do_align and landmarks is not None:
        aligned = align_face(frame_rgb, landmarks, 112)
        aligned = np.ascontiguousarray(aligned)
    else:
        face_roi = frame_rgb[y:y+h, x:x+w]
        aligned = np.ascontiguousarray(face_roi) if face_roi.size > 0 else None

    if aligned is None:
        return RecognizeResponse(status="error", message="Face alignment failed")

    # ── Step 6: Embed ──
    embedding = state.embedder.embed(aligned)
    if embedding is None:
        return RecognizeResponse(status="error", message="Embedding extraction failed")

    # ── Step 7: Match ──
    label, distance, confidence = find_best_match(
        embedding, state.known_encodings, state.known_labels, state.threshold
    )
    if label == "Unknown":
        state.pending.clear()
        return RecognizeResponse(
            status="unknown",
            distance=round(distance, 4),
            message="Face not recognized.",
        )

    # ── Step 8: Cooldown check ──
    now = time.time()
    last_seen = state.cooldown.get(label, 0)
    if now - last_seen < state.cooldown_seconds:
        remaining = int(state.cooldown_seconds - (now - last_seen))
        return RecognizeResponse(
            status="cooldown",
            identity=label,
            distance=round(distance, 4),
            message=f"Already recorded. Please wait {remaining}s.",
        )

    # ── Step 9: Identity consensus (confirm same person across N frames) ──
    consensus_required = state.consensus_required
    pending = state.pending.get(label)

    if pending is None or now - pending["last_time"] > 10.0:
        state.pending.clear()
        state.pending[label] = {"count": 1, "last_time": now, "distances": [distance]}
    else:
        pending["count"] += 1
        pending["last_time"] = now
        pending["distances"].append(distance)

    current_count = state.pending[label]["count"]
    print(f"[CONSENSUS] {label}: {current_count}/{consensus_required} "
          f"(dist={distance:.4f})")

    if current_count < consensus_required:
        return RecognizeResponse(
            status="verifying",
            identity=label,
            distance=round(distance, 4),
            consensus_progress=current_count,
            consensus_required=consensus_required,
            message=f"Verifying... ({current_count}/{consensus_required})",
        )

    # Consensus reached — clear pending, start liveness challenge
    state.pending.clear()

    # ── Step 11: Start liveness challenge ──
    session = state.liveness.start_session(label, landmarks, frame_rgb, distance)
    challenge_instructions = {
        "blink": "Please BLINK",
        "nod": "Please NOD slowly",
    }
    return RecognizeResponse(
        status="liveness_challenge",
        identity=label,
        distance=round(distance, 4),
        challenge_type=session.challenge_type.value,
        challenge_instruction=challenge_instructions[session.challenge_type.value],
        challenge_time_remaining=session.timeout,
        message=challenge_instructions[session.challenge_type.value],
    )


@app.post("/api/enroll")
async def enroll(request: Request, body: EnrollRequest):
    """Enroll a new employee via live camera capture."""
    state = request.app.state

    # ── PIN check ──
    if state.enrollment_pin and body.pin != state.enrollment_pin:
        return EnrollResponse(status="unauthorized", message="Invalid PIN.")

    # ── Validate and sanitize name (prevent path traversal) ──
    first = re.sub(r"[^a-zA-Z\s-]", "", body.first_name).strip().lower()
    last = re.sub(r"[^a-zA-Z\s-]", "", body.last_name).strip().lower()
    if not first or not last:
        return EnrollResponse(status="error", message="First and last name are required (letters, spaces, hyphens only).")

    first = re.sub(r"\s+", "_", first)
    last = re.sub(r"\s+", "_", last)
    base_name = f"{first}_{last}"

    # ── Resolve employee name (auto-suffix on collision) ──
    conn = state.db_conn
    inactive = conn.execute(
        "SELECT id FROM employees WHERE id = ? AND is_active = 0", (base_name,)
    ).fetchone()

    if inactive:
        # Same base name exists but is soft-deleted — this is a re-enrollment
        employee_name = base_name
    else:
        # Find all existing IDs (active + inactive) matching this base
        existing_ids = {
            row[0] for row in conn.execute(
                "SELECT id FROM employees WHERE id = ? OR id LIKE ?",
                (base_name, f"{base_name}_%"),
            ).fetchall()
        }
        existing_ids.update(state.known_labels)  # cover any pkl/SQLite drift
        employee_name = base_name
        suffix = 2
        while employee_name in existing_ids:
            employee_name = f"{base_name}_{suffix}"
            suffix += 1

    # ── Decode image ──
    try:
        image_data = body.image
        if "," in image_data and image_data.index(",") < 100:
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return EnrollResponse(status="error", message="Failed to decode image.", employee_name=employee_name)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return EnrollResponse(status="error", message=f"Image decode failed: {e}", employee_name=employee_name)

    # ── Detect face ──
    detections = state.detector.detect(frame_rgb)
    if len(detections) == 0:
        return EnrollResponse(status="no_face", message="No face detected — try again.", employee_name=employee_name)
    if len(detections) > 1:
        return EnrollResponse(
            status="multiple_faces",
            message="Multiple faces detected — only you should be in frame.",
            employee_name=employee_name,
        )

    (x, y, w, h), landmarks = detections[0]

    # ── Anti-spoof check ──
    if state.anti_spoof is not None:
        face_crop = frame_rgb[y:y+h, x:x+w]
        if face_crop.size > 0:
            is_real, spoof_score = state.anti_spoof.check(face_crop)
            if not is_real:
                return EnrollResponse(
                    status="spoof_detected",
                    message="Liveness check failed — use your real face.",
                    employee_name=employee_name,
                )

    # ── Align face ──
    if state.do_align and landmarks is not None:
        aligned = align_face(frame_rgb, landmarks, 112)
        aligned = np.ascontiguousarray(aligned)
    else:
        face_roi = frame_rgb[y:y+h, x:x+w]
        aligned = np.ascontiguousarray(face_roi) if face_roi.size > 0 else None

    if aligned is None:
        return EnrollResponse(status="error", message="Face alignment failed.", employee_name=employee_name)

    # ── Embed ──
    embedding = state.embedder.embed(aligned)
    if embedding is None:
        return EnrollResponse(status="error", message="Embedding extraction failed.", employee_name=employee_name)

    # ── Persist to pkl (before in-memory and photo — safer ordering) ──
    try:
        save_encoding_to_pkl(args.database, embedding, employee_name)
    except Exception as e:
        return EnrollResponse(status="error", message=f"Failed to save encoding: {e}", employee_name=employee_name)

    # ── Save photo (after pkl — no orphan files on failure) ──
    employees_dir = Path(args.database).parent / "employees"
    employees_dir.mkdir(parents=True, exist_ok=True)
    photo_file = employees_dir / f"{employee_name}.jpg"
    cv2.imwrite(str(photo_file), frame_bgr)

    # ── Hot-reload in-memory ──
    state.known_encodings.append(embedding.tolist())
    state.known_labels.append(employee_name)

    # ── Insert into employees table ──
    ts = datetime.now(timezone.utc).isoformat()
    state.db_conn.execute(
        "INSERT INTO employees (id, name, enrolled_at, photo_path, is_active, store_id) "
        "VALUES (?, ?, ?, ?, 1, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "is_active = 1, enrolled_at = excluded.enrolled_at, "
        "photo_path = excluded.photo_path, store_id = excluded.store_id",
        (employee_name, f"{first} {last}", ts, str(photo_file), state.store_id),
    )
    state.db_conn.commit()

    print(f"[ENROLL] {employee_name} enrolled successfully")
    return EnrollResponse(
        status="enrolled",
        message=f"{employee_name} enrolled successfully!",
        employee_name=employee_name,
    )


@app.delete("/api/enroll/{employee_id}")
async def delete_employee(request: Request, employee_id: str, pin: Optional[str] = Query(default=None)):
    """Remove an employee from recognition and mark as inactive."""
    state = request.app.state

    # ── PIN check ──
    if state.enrollment_pin and pin != state.enrollment_pin:
        return {"status": "unauthorized", "message": "Invalid PIN.", "employee_id": employee_id}

    # ── Check exists and is active ──
    row = state.db_conn.execute(
        "SELECT photo_path FROM employees WHERE id = ? AND is_active = 1",
        (employee_id,),
    ).fetchone()
    if row is None:
        return {"status": "not_found", "message": f"No active employee found with id '{employee_id}'.", "employee_id": employee_id}

    photo_path = row[0]

    # ── Remove from pkl (before in-memory — safer ordering) ──
    try:
        remove_encoding_from_pkl(args.database, employee_id)
    except Exception as e:
        return {"status": "error", "message": f"Failed to update encoding database: {e}", "employee_id": employee_id}

    # ── Hot-reload in-memory: filter both parallel lists ──
    pairs = [(enc, lbl) for enc, lbl in zip(state.known_encodings, state.known_labels) if lbl != employee_id]
    if pairs:
        state.known_encodings, state.known_labels = map(list, zip(*pairs))
    else:
        state.known_encodings, state.known_labels = [], []

    # ── Soft-delete in SQLite ──
    deactivate_employee(state.db_conn, employee_id)

    # ── Delete photo file ──
    if photo_path:
        try:
            Path(photo_path).unlink(missing_ok=True)
        except Exception:
            pass  # Non-fatal — record is already deactivated

    # ── Clear in-memory state for this identity ──
    state.cooldown.pop(employee_id, None)
    state.pending.pop(employee_id, None)

    print(f"[DELETE] {employee_id} removed")
    return {"status": "deleted", "message": f"{employee_id} removed successfully.", "employee_id": employee_id}


@app.get("/api/employee/{employee_id}/photo")
async def get_employee_photo(request: Request, employee_id: str, pin: Optional[str] = Query(default=None)):
    """Serve the enrollment photo for an employee. PIN-gated."""
    state = request.app.state

    if state.enrollment_pin and pin != state.enrollment_pin:
        return JSONResponse({"status": "unauthorized", "message": "Invalid PIN."}, status_code=401)

    row = state.db_conn.execute(
        "SELECT photo_path FROM employees WHERE id = ? AND store_id = ?",
        (employee_id, state.store_id),
    ).fetchone()
    if row is None or not row[0]:
        return JSONResponse({"status": "not_found", "message": "Photo not found."}, status_code=404)

    photo_path = Path(row[0])
    if not photo_path.exists():
        return JSONResponse({"status": "not_found", "message": "Photo file missing on disk."}, status_code=404)

    return FileResponse(str(photo_path), media_type="image/jpeg")


@app.get("/api/employees")
async def list_employees(request: Request):
    """Return all active enrolled employees for this store."""
    state = request.app.state
    employees = get_active_employees(state.db_conn, state.store_id)
    for emp in employees:
        photo_path = emp.pop("photo_path")
        emp["has_photo"] = bool(photo_path and Path(photo_path).exists())
    return {"employees": employees, "count": len(employees)}


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
