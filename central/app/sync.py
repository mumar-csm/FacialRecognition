"""POST /api/sync/batch — receive event bundles from kiosks.

The kiosk's sync_worker drains its local outbox in batches and POSTs them here.
Each event has an event_uuid (idempotency key), a kind, and a payload dict.

Per-event isolation: every event runs in its own DB transaction so one bad
event can't poison the batch. Validation failures (store_id mismatch, unknown
kind, missing field, FK violation) are logged to sync_audit and returned in
the response's skipped[] array; we still return 200 so the kiosk drops the
poison from its outbox.

Unexpected errors (DB connection drop, etc.) bubble up to FastAPI as 500,
which causes the kiosk to retry the entire batch on its next tick.
"""

import base64
import binascii
import json
from datetime import datetime
from typing import Any, Optional

from asyncpg.exceptions import ForeignKeyViolationError
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from . import db
from .auth import DeviceAuth, require_device
from .models import attendance, deletion_audit, employees, spoof_attempts, sync_audit


router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────
# We only validate the outer envelope. Payload is dict[str, Any] because each
# kind has different fields — handlers fish out what they need and a KeyError
# becomes a malformed_payload audit.

class EventIn(BaseModel):
    event_uuid: str
    kind: str
    payload: dict[str, Any]


class BatchIn(BaseModel):
    events: list[EventIn]


# ── Helpers ────────────────────────────────────────────────────────────────

_PAYLOAD_PREVIEW_LIMIT = 500


def _ts(s: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a datetime.

    The kiosk emits timestamps as `datetime.now(timezone.utc).isoformat()` (see
    kiosk_server.py). Parsing here so we hand asyncpg a datetime rather than
    relying on per-driver string coercion, and so malformed timestamps surface
    as ValueError → malformed_payload audit.
    """
    return datetime.fromisoformat(s)


async def _write_audit(
    session: AsyncSession,
    auth: DeviceAuth,
    event: EventIn,
    reason: str,
) -> None:
    """Insert one sync_audit row. Caller manages the transaction."""
    preview = json.dumps(event.payload)[:_PAYLOAD_PREVIEW_LIMIT]
    await session.execute(
        sync_audit.insert().values(
            device_id=auth.device_id,
            store_id=event.payload.get("store_id") if isinstance(event.payload, dict) else None,
            event_uuid=event.event_uuid,
            kind=event.kind,
            reason=reason,
            payload_preview=preview,
        )
    )


# ── Per-kind handlers ──────────────────────────────────────────────────────

async def _handle_attendance(
    session: AsyncSession, event: EventIn, auth: DeviceAuth
) -> None:
    p = event.payload
    await session.execute(
        pg_insert(attendance)
        .values(
            event_uuid=event.event_uuid,
            store_id=p["store_id"],
            device_id=p["device_id"],
            timestamp=_ts(p["timestamp"]),
            employee_id=p["employee_id"],
            distance=float(p["distance"]),
            is_clock_in=bool(p["is_clock_in"]),
            camera_id=p["camera_id"],
        )
        .on_conflict_do_nothing(index_elements=["event_uuid"])
    )


async def _handle_spoof_attempt(
    session: AsyncSession, event: EventIn, auth: DeviceAuth
) -> None:
    p = event.payload
    await session.execute(
        pg_insert(spoof_attempts)
        .values(
            event_uuid=event.event_uuid,
            store_id=p["store_id"],
            device_id=p["device_id"],
            timestamp=_ts(p["timestamp"]),
            camera_id=p["camera_id"],
            spoof_score=float(p["spoof_score"]),
        )
        .on_conflict_do_nothing(index_elements=["event_uuid"])
    )


async def _handle_enrollment(
    session: AsyncSession, event: EventIn, auth: DeviceAuth
) -> None:
    p = event.payload
    encoding_bytes = base64.b64decode(p["encoding_b64"], validate=True)
    photo_b64 = p.get("photo_b64") or ""
    photo_bytes = base64.b64decode(photo_b64, validate=True) if photo_b64 else None
    enrolled_at = _ts(p["timestamp"])

    # pos_employee_id is nullable on the column (pre-migration rows) but the
    # kiosk requires it for new enrollments. Treat missing/empty as None here
    # for forward-compat with any older kiosk payloads still in flight.
    pos_employee_id = p.get("pos_employee_id") or None

    stmt = pg_insert(employees).values(
        id=p["employee_id"],
        store_id=p["store_id"],
        display_name=p["display_name"],
        enrolled_at=enrolled_at,
        is_active=True,
        embedder_type=p["embedder_type"],
        embedding_dim=int(p["embedding_dim"]),
        encoding=encoding_bytes,
        photo=photo_bytes,
        pos_employee_id=pos_employee_id,
    )
    # Newer enrollment wins; equal-timestamp duplicate retries are no-ops
    # (avoids version churn). See conversation notes 2026-05-19.
    stmt = stmt.on_conflict_do_update(
        index_elements=["id", "store_id"],
        set_={
            "display_name": stmt.excluded.display_name,
            "enrolled_at": stmt.excluded.enrolled_at,
            "is_active": True,
            "embedder_type": stmt.excluded.embedder_type,
            "embedding_dim": stmt.excluded.embedding_dim,
            "encoding": stmt.excluded.encoding,
            "photo": stmt.excluded.photo,
            "pos_employee_id": stmt.excluded.pos_employee_id,
            # Draw a fresh store-monotonic version from the sequence (migration
            # 0006) so the roster watermark advances correctly. A bare +1 would
            # repeat values across employees and let the roster pull miss
            # changes. New-row inserts get nextval via the column default.
            "version": func.nextval("employee_change_seq"),
            "updated_at": func.now(),
        },
        where=stmt.excluded.enrolled_at > employees.c.enrolled_at,
    )
    await session.execute(stmt)


async def _handle_deactivation(
    session: AsyncSession, event: EventIn, auth: DeviceAuth
) -> None:
    """Deactivate an employee AND erase their biometric.

    Beyond flipping is_active, this wipes the encoding (face template) and photo
    to NULL — mirroring the kiosk, which already removes the encoding from its
    pkl and deletes the enrollment photo on delete. The row, display_name,
    pos_employee_id and all attendance history are kept so payroll/audit reports
    still resolve. A later re-enrollment (newer timestamp) restores a fresh
    encoding via _handle_enrollment.
    """
    p = event.payload
    incoming_ts = _ts(p["timestamp"])
    # Gate against stale events: incoming kiosk timestamp must be >= the last
    # time any source modified this row. Today this only catches same-kiosk
    # replays (e.g. outbox redelivery). Once Step 2b adds HQ-side admin
    # endpoints, the same gate will also stop a queued kiosk event from
    # clobbering an HQ reactivation. That's why we compare against updated_at
    # (server clock, bumped by every writer) rather than enrolled_at (kiosk
    # clock only) — single comparison protects both kiosk-vs-kiosk and
    # kiosk-vs-HQ ordering. The residual cost is kiosk-clock-skew: if a Pi's
    # clock is behind central's by more than the gap between an enrollment
    # and a follow-up deactivation, the deactivation gets wrongly rejected.
    # Likelihood is low (NTP normally <1s drift, deactivations rarely follow
    # enrollments by seconds) and failure mode is recoverable (admin retries).
    # No-op (0 rows updated) is fine — naturally idempotent.
    result = await session.execute(
        update(employees)
        .where(employees.c.id == p["employee_id"])
        .where(employees.c.store_id == p["store_id"])
        .where(employees.c.updated_at <= incoming_ts)
        .values(
            is_active=False,
            # Erase the biometric: face template + enrollment photo gone, while
            # the row and attendance history stay (encoding is nullable as of
            # migration 0004; photo was already nullable).
            encoding=None,
            photo=None,
            updated_at=func.now(),
            # Store-monotonic version from the sequence (see migration 0006 and
            # the enrollment handler) — the roster watermark relies on versions
            # being unique + increasing across the whole store.
            version=func.nextval("employee_change_seq"),
        )
    )

    # Record the erasure ONLY when it actually applied. A stale/no-op redelivery
    # (gate rejected it, or the employee isn't here) updates 0 rows and must not
    # leave a phantom audit entry. Same transaction as the UPDATE (the caller
    # opened it), so erasure and audit commit or roll back together.
    if result.rowcount:
        await session.execute(
            deletion_audit.insert().values(
                device_id=auth.device_id,
                store_id=p["store_id"],
                employee_id=p["employee_id"],
                event_uuid=event.event_uuid,
                event_timestamp=incoming_ts,
            )
        )


HANDLERS = {
    "attendance": _handle_attendance,
    "spoof_attempt": _handle_spoof_attempt,
    "enrollment": _handle_enrollment,
    "deactivation": _handle_deactivation,
}


# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post("/api/sync/batch")
async def sync_batch(
    body: BatchIn,
    auth: DeviceAuth = Depends(require_device),
) -> dict[str, Any]:
    processed = 0
    skipped: list[dict[str, Any]] = []

    async with db.sessionmaker()() as session:
        for event in body.events:
            reason = await _process_one(session, event, auth)
            if reason is None:
                processed += 1
            else:
                skipped.append({
                    "event_uuid": event.event_uuid,
                    "kind": event.kind,
                    "reason": reason,
                })

    return {"processed": processed, "skipped": skipped}


async def _process_one(
    session: AsyncSession,
    event: EventIn,
    auth: DeviceAuth,
) -> Optional[str]:
    """Process one event. Returns None on success, or a skip-reason string.

    On expected validation failures (mismatch, unknown kind, FK, malformed),
    write an audit row in its own transaction and return the reason. On
    unexpected errors, re-raise — the batch endpoint will turn it into a 500
    and the kiosk will retry the whole batch.
    """
    # 1. store_id sanity — payload must claim the same store as the API key.
    claimed_store_id = event.payload.get("store_id") if isinstance(event.payload, dict) else None
    if claimed_store_id != auth.store_id:
        async with session.begin():
            await _write_audit(session, auth, event, "store_id_mismatch")
        return "store_id_mismatch"

    # 2. Dispatch by kind.
    handler = HANDLERS.get(event.kind)
    if handler is None:
        async with session.begin():
            await _write_audit(session, auth, event, "unknown_kind")
        return "unknown_kind"

    # 3. Run the handler in its own tx. Validation-class errors get caught
    #    and audited; transient/unknown errors bubble up to a 500.
    try:
        async with session.begin():
            await handler(session, event, auth)
        return None
    except IntegrityError as e:
        # Most likely FK violation on attendance.employee_id (employee not yet
        # enrolled). Anything else under IntegrityError is unexpected but
        # treating it as a per-event skip is safer than poisoning the batch.
        reason = "fk_violation" if isinstance(e.orig, ForeignKeyViolationError) else "integrity_error"
        async with session.begin():
            await _write_audit(session, auth, event, reason)
        return reason
    except (KeyError, ValueError, TypeError, binascii.Error):
        async with session.begin():
            await _write_audit(session, auth, event, "malformed_payload")
        return "malformed_payload"
