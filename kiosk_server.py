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
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
from sync_worker import SyncWorker


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
            id               TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            enrolled_at      TEXT NOT NULL,
            photo_path       TEXT,
            is_active        INTEGER NOT NULL DEFAULT 1,
            store_id         TEXT NOT NULL DEFAULT 'store-01',
            pos_employee_id  TEXT CHECK (
                pos_employee_id IS NULL
                OR pos_employee_id GLOB '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
            )
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

        CREATE TABLE IF NOT EXISTS outbox (
            event_uuid    TEXT PRIMARY KEY,
            kind          TEXT NOT NULL,
            payload_json  TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            sent_at       TEXT,
            attempts      INTEGER NOT NULL DEFAULT 0,
            last_error    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp);
        CREATE INDEX IF NOT EXISTS idx_attendance_employee  ON attendance(employee_id);
        CREATE INDEX IF NOT EXISTS idx_spoof_timestamp      ON spoof_attempts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_outbox_unsent ON outbox(created_at) WHERE sent_at IS NULL;

        -- Reject NULL pos_employee_id on new inserts. Pre-migration rows already
        -- exist with NULL and are untouched (this is BEFORE INSERT, not UPDATE),
        -- but every new employee must have a POS ID. Backstops the API regex.
        CREATE TRIGGER IF NOT EXISTS trg_employees_pos_id_required_on_insert
        BEFORE INSERT ON employees
        WHEN NEW.pos_employee_id IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'pos_employee_id is required for new employees');
        END;
    """)

    # Migration: add store_id to existing databases that predate this column
    try:
        conn.execute("ALTER TABLE employees ADD COLUMN store_id TEXT NOT NULL DEFAULT 'store-01'")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: add pos_employee_id (nullable so pre-existing rows don't fail).
    # The /api/enroll endpoint requires it for new enrollments — see EnrollRequest.
    try:
        conn.execute("ALTER TABLE employees ADD COLUMN pos_employee_id TEXT")
    except sqlite3.OperationalError:
        pass

    # Partial unique index: two employees in the same store cannot share a POS
    # ID (Oracle uses it as the employee identifier on punches). Multiple NULLs
    # are fine — pre-migration rows aren't blocked.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uniq_employees_store_pos "
        "ON employees(store_id, pos_employee_id) "
        "WHERE pos_employee_id IS NOT NULL"
    )

    conn.commit()
    return conn


def _enqueue_outbox(conn: sqlite3.Connection, kind: str,
                    payload: Dict[str, Any], created_at: str) -> str:
    """Insert an outbox row. Must be called inside an open transaction.

    Returns the event_uuid so callers can correlate logs.
    """
    event_uuid = payload.setdefault("event_uuid", str(uuid.uuid4()))
    conn.execute(
        "INSERT INTO outbox (event_uuid, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (event_uuid, kind, json.dumps(payload, separators=(",", ":")), created_at),
    )
    return event_uuid


def log_attendance(conn: sqlite3.Connection, employee_id: str,
                   distance: float, is_clock_in: bool,
                   store_id: str, device_id: str,
                   camera_id: str = "kiosk-01") -> int:
    """Insert an attendance record + matching outbox row in one transaction.

    Returns the inserted attendance row ID. The outbox row is what the sync
    worker drains up to central; same transaction means we can't observe an
    attendance row that hasn't been queued for upload.
    """
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "store_id": store_id,
        "device_id": device_id,
        "timestamp": ts,
        "employee_id": employee_id,
        "distance": distance,
        "is_clock_in": bool(is_clock_in),
        "camera_id": camera_id,
    }
    with conn:
        cursor = conn.execute(
            "INSERT INTO attendance (timestamp, employee_id, distance, is_clock_in, camera_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, employee_id, distance, int(is_clock_in), camera_id),
        )
        _enqueue_outbox(conn, "attendance", payload, ts)
    return cursor.lastrowid


def log_spoof_attempt(conn: sqlite3.Connection, camera_id: str, spoof_score: float,
                      store_id: str, device_id: str) -> int:
    """Insert a spoof attempt record + matching outbox row in one transaction."""
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "store_id": store_id,
        "device_id": device_id,
        "timestamp": ts,
        "camera_id": camera_id,
        "spoof_score": spoof_score,
    }
    with conn:
        cursor = conn.execute(
            "INSERT INTO spoof_attempts (timestamp, camera_id, spoof_score) VALUES (?, ?, ?)",
            (ts, camera_id, spoof_score),
        )
        _enqueue_outbox(conn, "spoof_attempt", payload, ts)
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


def get_active_employees(conn: sqlite3.Connection, store_id: str) -> List[Dict]:
    """Return all active enrolled employees for a given store."""
    rows = conn.execute(
        "SELECT id, name, enrolled_at, photo_path, pos_employee_id FROM employees "
        "WHERE is_active = 1 AND store_id = ? ORDER BY name",
        (store_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "enrolled_at": r[2],
            "photo_path": r[3],
            "pos_employee_id": r[4],
        }
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


def reconcile_recognition_state(
    conn: sqlite3.Connection,
    db_path: str,
    known_encodings: list,
    known_labels: list,
) -> Tuple[list, list, int]:
    """Heal pkl/in-memory drift against the SQLite source of truth at startup.

    A delete records the soft-delete + central notification BEFORE it wipes the
    pkl (see delete_employee). If the process dies in that window, the face is
    still present in the pkl and the loaded in-memory lists even though the
    employee row is already marked inactive. On startup we drop any recognition
    entry whose employee is soft-deleted, so a crashed delete self-completes,
    and we persist the heal back to the pkl so disk matches memory.

    SQLite is authoritative here because the soft-delete is the durable record
    of intent; the pkl is only the recognition substrate. Returns the
    (possibly filtered) (encodings, labels) and the count of entries removed.
    """
    inactive = {
        r[0] for r in conn.execute("SELECT id FROM employees WHERE is_active = 0")
    }
    stale = inactive.intersection(known_labels)
    if not stale:
        return known_encodings, known_labels, 0

    # Persist the heal first (best-effort): drop each stale label from the pkl
    # so the reclaimed biometric doesn't linger on disk. A failure here is
    # non-fatal — the in-memory filter below still takes effect this run, and
    # the next startup will retry the pkl rewrite.
    for label in stale:
        try:
            remove_encoding_from_pkl(db_path, label)
        except Exception as e:
            print(f"[STARTUP] WARN: could not purge '{label}' from pkl during reconcile: {e}")

    pairs = [(e, l) for e, l in zip(known_encodings, known_labels) if l not in stale]
    if pairs:
        known_encodings, known_labels = map(list, zip(*pairs))
    else:
        known_encodings, known_labels = [], []
    return known_encodings, known_labels, len(stale)


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
    p.add_argument("--store-id", default=None,
                   dest="store_id",
                   help="Identifier for this store location (e.g. 'downtown-01'). "
                        "Falls back to --device-config, then to 'store-01'.")
    p.add_argument("--device-id", default=None, dest="device_id",
                   help="Stable identifier for this Pi (e.g. 'pi-store-01-a'). "
                        "Falls back to --device-config, then to the machine hostname.")
    p.add_argument("--device-config", default=None, dest="device_config",
                   help="Optional JSON file with device_id/store_id/central_url/sync_interval_seconds/sync_batch_size. "
                        "CLI flags override individual fields. The API key is read from CENTRAL_API_KEY env var only.")
    p.add_argument("--central-url", default=None, dest="central_url",
                   help="Base URL of the central HQ server (e.g. https://central.example.com). "
                        "If omitted, outbox rows are written but never drained — useful for dev/test.")
    p.add_argument("--sync-interval-seconds", type=int, default=None,
                   dest="sync_interval_seconds",
                   help="How often the sync worker drains the outbox (seconds, production default 1800 = 30 min; "
                        "override to 30 for dev/testing)")
    p.add_argument("--sync-batch-size", type=int, default=None,
                   dest="sync_batch_size",
                   help="Max outbox rows per upload batch (default 50)")
    p.add_argument("--enrollment-pin", default=None,
                   dest="enrollment_pin",
                   help="Manager PIN required to enroll/delete employees (optional)")
    p.add_argument("--timezone", default="UTC",
                   help="Local timezone for report timestamps (e.g. 'America/New_York')")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Defaults to loopback so the API is only reachable from the same machine "
                        "(the Pi-kiosk model). Set to '0.0.0.0' only if you intentionally want LAN exposure.")
    p.add_argument("--port", type=int, default=8000)
    return _apply_device_config(p.parse_args())


def _apply_device_config(args):
    """Layer values from --device-config under the CLI flags, then apply final defaults.

    Precedence: CLI flag > config file > hard default. The API key is intentionally
    *not* read from the config file — it lives in CENTRAL_API_KEY env var only, so
    it stays out of `ps aux` and out of any config file that might get committed.
    """
    cfg: Dict[str, Any] = {}
    if args.device_config:
        try:
            with open(args.device_config) as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read --device-config {args.device_config}: {e}")
            sys.exit(1)

    # Fields that can come from CLI or config file
    for field in ("device_id", "store_id", "central_url",
                  "sync_interval_seconds", "sync_batch_size"):
        if getattr(args, field, None) is None and field in cfg:
            setattr(args, field, cfg[field])

    # Hard defaults — applied last so config-file values still win
    if args.store_id is None:
        args.store_id = "store-01"
    if args.device_id is None:
        import socket
        args.device_id = socket.gethostname() or "unknown-device"
    if args.sync_interval_seconds is None:
        args.sync_interval_seconds = 1800  # 30 min production default. Override to 30 for dev/testing.
    if args.sync_batch_size is None:
        args.sync_batch_size = 50

    # Fail fast: enabling sync requires both a URL and an API key
    if args.central_url:
        if not os.environ.get("CENTRAL_API_KEY"):
            print("[ERROR] --central-url is set but CENTRAL_API_KEY env var is empty. "
                  "Refusing to start: the kiosk would write outbox rows that can't be uploaded.")
            sys.exit(1)

    return args


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

    # Face encodings + SQLite. Open the DB before binding the in-memory lists:
    # SQLite is the source of truth for who's deleted, so we reconcile the pkl
    # against it first (heals a delete that crashed before its pkl wipe).
    encodings, labels = load_database(args.database, expected_embedder=args.embedder)
    app.state.db_conn = init_kiosk_db(args.sqlite)

    encodings, labels, healed = reconcile_recognition_state(
        app.state.db_conn, args.database, encodings, labels
    )
    if healed:
        print(f"[STARTUP] Reconciled {healed} stale recognition "
              f"entr{'y' if healed == 1 else 'ies'} against soft-deleted employees")
    app.state.known_encodings = encodings
    app.state.known_labels = labels

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
    app.state.device_id = args.device_id
    app.state.enrollment_pin = args.enrollment_pin
    app.state.local_tz = local_tz

    pin_status = "protected" if args.enrollment_pin else "open"
    sync_status = (f"central={args.central_url}, interval={args.sync_interval_seconds}s, "
                   f"batch={args.sync_batch_size}") if args.central_url else "disabled"
    print(f"[STARTUP] Ready — store={args.store_id}, device={args.device_id}, "
          f"tz={args.timezone}, enrollment={pin_status}, {len(labels)} employees, "
          f"threshold={args.threshold}, cooldown={args.cooldown}s, "
          f"consensus={args.consensus} frames, spoof_threshold={args.spoof_threshold}, "
          f"challenge_timeout={args.challenge_timeout}s, sync={sync_status}")

    # Start the outbox sync worker if a central URL is configured. When it isn't,
    # outbox rows still get written (kiosk works offline) — they just sit until
    # the operator wires up --central-url.
    app.state.sync_worker = None
    if args.central_url:
        app.state.sync_worker = SyncWorker(
            db_path=args.sqlite,
            central_url=args.central_url,
            api_key=os.environ["CENTRAL_API_KEY"],
            interval_seconds=args.sync_interval_seconds,
            batch_size=args.sync_batch_size,
        )
        app.state.sync_worker.start()

    yield

    # Shutdown
    if app.state.sync_worker is not None:
        await app.state.sync_worker.stop()
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
    pos_employee_id: str   # Oracle POS employee identifier — used to map punches
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
    start = time.perf_counter()
    try:
        return await _recognize_impl(request, body)
    finally:
        print(f"recognize took {(time.perf_counter() - start) * 1000:.1f} ms", flush=True)


async def _recognize_impl(request: Request, body: RecognizeRequest):
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
            log_spoof_attempt(state.db_conn, camera_id, 0.0,
                              store_id=state.store_id, device_id=state.device_id)
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
        log_attendance(state.db_conn, active_identity, avg_distance, is_clock_in,
                       store_id=state.store_id, device_id=state.device_id, camera_id=camera_id)
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
                    log_spoof_attempt(state.db_conn, camera_id, spoof_score,
                                      store_id=state.store_id, device_id=state.device_id)
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

    # ── Validate POS employee ID ──
    # Oracle Simphony tenants on this deployment issue 7-digit numeric IDs.
    pos_id = body.pos_employee_id.strip()
    if not re.fullmatch(r"\d{7}", pos_id):
        return EnrollResponse(
            status="error",
            message="POS Employee ID must be exactly 7 digits.",
        )

    # Uniqueness within store — POS ID identifies the employee on Oracle punches,
    # so two active employees can't share it. Soft-deleted rows are excluded so
    # an inactive employee's old POS ID can be reused (or kept on re-enrollment
    # of the same name, which overwrites the row by id below).
    conn = state.db_conn
    pos_collision = conn.execute(
        "SELECT id FROM employees "
        "WHERE store_id = ? AND pos_employee_id = ? AND is_active = 1",
        (state.store_id, pos_id),
    ).fetchone()
    if pos_collision:
        return EnrollResponse(
            status="error",
            message=f"POS Employee ID '{pos_id}' is already in use by another employee.",
        )

    # ── Resolve employee name (auto-suffix on collision) ──
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

    # ── Insert into employees table + queue enrollment event for central ──
    ts = datetime.now(timezone.utc).isoformat()
    ok, photo_jpg = cv2.imencode(".jpg", frame_bgr)
    photo_b64 = base64.b64encode(photo_jpg.tobytes()).decode("ascii") if ok else ""
    encoding_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
    enrollment_payload = {
        "store_id": state.store_id,
        "device_id": state.device_id,
        "timestamp": ts,
        "employee_id": employee_name,
        "display_name": f"{first} {last}",
        "pos_employee_id": pos_id,
        "embedder_type": args.embedder,
        "embedding_dim": int(np.asarray(embedding).size),
        "encoding_b64": base64.b64encode(encoding_bytes).decode("ascii"),
        "photo_b64": photo_b64,
    }
    with state.db_conn:
        state.db_conn.execute(
            "INSERT INTO employees (id, name, enrolled_at, photo_path, is_active, store_id, pos_employee_id) "
            "VALUES (?, ?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "is_active = 1, enrolled_at = excluded.enrolled_at, "
            "photo_path = excluded.photo_path, store_id = excluded.store_id, "
            "pos_employee_id = excluded.pos_employee_id",
            (employee_name, f"{first} {last}", ts, str(photo_file), state.store_id, pos_id),
        )
        _enqueue_outbox(state.db_conn, "enrollment", enrollment_payload, ts)

    print(f"[ENROLL] {employee_name} enrolled successfully")
    return EnrollResponse(
        status="enrolled",
        message=f"{employee_name} enrolled successfully!",
        employee_name=employee_name,
    )


@app.delete("/api/enroll/{employee_id}")
async def delete_employee(request: Request, employee_id: str, pin: Optional[str] = Query(default=None)):
    """Remove an employee from recognition and mark as inactive.

    Failure-aware ordering. The durable record of intent — the SQLite
    soft-delete plus the `deactivation` outbox event that notifies central — is
    committed FIRST, before the irreversible pkl wipe. So if the process dies
    mid-delete, the worst residual is "row inactive + central notified, but face
    still in the pkl", which is self-healed on next startup by
    reconcile_recognition_state() and also fixable by simply retrying this call.
    The old order (pkl wipe first) could strand the system as "face gone but row
    still active and central never told" — a silent, unrecoverable drift.

    Idempotent: an already-inactive row skips the soft-delete/outbox step (no
    double-notify) but still re-runs the pkl/in-memory/photo cleanup, so a retry
    completes a delete that previously failed after the soft-delete commit.
    """
    state = request.app.state

    # ── PIN check ──
    if state.enrollment_pin and pin != state.enrollment_pin:
        return {"status": "unauthorized", "message": "Invalid PIN.", "employee_id": employee_id}

    # ── Look up the row regardless of is_active, so a retry can finish a
    #    partially-applied delete (an active-only filter would 404 the retry). ──
    row = state.db_conn.execute(
        "SELECT photo_path, is_active FROM employees WHERE id = ?",
        (employee_id,),
    ).fetchone()
    if row is None:
        return {"status": "not_found", "message": f"No employee found with id '{employee_id}'.", "employee_id": employee_id}

    photo_path, is_active = row[0], row[1]

    # ── 1. Durable intent FIRST: soft-delete + queue deactivation (one txn). ──
    #    Only on the active→inactive transition, so retries don't churn the
    #    outbox or double-notify central. Nothing irreversible has happened yet,
    #    so a failure here is safe to bail on and retry.
    if is_active:
        ts = datetime.now(timezone.utc).isoformat()
        deactivation_payload = {
            "store_id": state.store_id,
            "device_id": state.device_id,
            "timestamp": ts,
            "employee_id": employee_id,
        }
        try:
            with state.db_conn:
                state.db_conn.execute(
                    "UPDATE employees SET is_active = 0 WHERE id = ? AND is_active = 1",
                    (employee_id,),
                )
                _enqueue_outbox(state.db_conn, "deactivation", deactivation_payload, ts)
        except Exception as e:
            return {"status": "error", "message": f"Failed to record deletion: {e}", "employee_id": employee_id}

    # ── 2. Irreversible pkl wipe (persists the removal across restart). ──
    #    If this throws, the row is already inactive and central is queued; the
    #    residual face in the pkl is healed by reconcile on next startup or by
    #    retrying this endpoint.
    try:
        remove_encoding_from_pkl(args.database, employee_id)
    except Exception as e:
        return {"status": "error", "message": f"Marked inactive, but failed to update encoding database — retry to finish: {e}", "employee_id": employee_id}

    # ── 3. Stop runtime recognition now: filter the in-memory lists. ──
    pairs = [(enc, lbl) for enc, lbl in zip(state.known_encodings, state.known_labels) if lbl != employee_id]
    if pairs:
        state.known_encodings, state.known_labels = map(list, zip(*pairs))
    else:
        state.known_encodings, state.known_labels = [], []

    # ── 4. Delete photo file (non-fatal — record is already deactivated). ──
    if photo_path:
        try:
            Path(photo_path).unlink(missing_ok=True)
        except Exception:
            pass

    # ── 5. Clear in-memory state for this identity. ──
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
