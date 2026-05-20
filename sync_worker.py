#!/usr/bin/env python3
"""
Sync worker — drains the local SQLite `outbox` table to central HQ.

Runs as a background asyncio task inside kiosk_server.py's FastAPI lifespan.
On each tick it:
  1. SELECTs a batch of unsent outbox rows (ordered by created_at).
  2. POSTs them to {central_url}/api/sync/batch with Bearer auth.
  3. On HTTP 2xx — marks all rows in the batch as sent.
  4. On any other outcome — increments `attempts` and records `last_error`.

Rows whose attempts exceed `MAX_ATTEMPTS` are quarantined: they're skipped on
future drains and logged loudly. This prevents one poison-pill event from
blocking the rest of the queue.

The worker is fully passive when --central-url is not set: kiosk_server.py
simply doesn't start it, and outbox rows accumulate locally for later sync
(e.g. after the operator configures the URL via --device-config).
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx


MAX_ATTEMPTS = 100  # Quarantine an event after this many failed uploads.


class SyncWorker:
    """Background drain loop for the local outbox table."""

    def __init__(
        self,
        db_path: str,
        central_url: str,
        api_key: str,
        interval_seconds: int = 30,
        batch_size: int = 50,
        request_timeout_seconds: float = 15.0,
    ):
        self.db_path = db_path
        self.batch_url = central_url.rstrip("/") + "/api/sync/batch"
        self.api_key = api_key
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.request_timeout_seconds = request_timeout_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="sync_worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=self.request_timeout_seconds + 5)
        except asyncio.TimeoutError:
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        # One client for the lifetime of the worker — connection reuse + faster retries.
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            print(f"[SYNC] worker started — target={self.batch_url}, "
                  f"interval={self.interval_seconds}s, batch={self.batch_size}")
            while not self._stop_event.is_set():
                try:
                    drained = await self._drain_once(client)
                    # If we drained a full batch, immediately try again — there may
                    # be more queued rows and we don't want to wait an interval per batch.
                    if drained == self.batch_size:
                        continue
                except Exception as e:
                    # Defensive: never let a worker bug crash the server.
                    print(f"[SYNC] tick failed unexpectedly: {e!r}")
                # Sleep, but wake immediately on stop.
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    pass
            print("[SYNC] worker stopped")

    async def _drain_once(self, client: httpx.AsyncClient) -> int:
        """Drain at most one batch. Returns number of rows attempted."""
        rows = self._select_batch()
        if not rows:
            return 0

        events = [
            {"event_uuid": r["event_uuid"], "kind": r["kind"], "payload": json.loads(r["payload_json"])}
            for r in rows
        ]
        event_uuids = [r["event_uuid"] for r in rows]

        try:
            resp = await client.post(
                self.batch_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"events": events},
            )
        except httpx.HTTPError as e:
            self._mark_failed(event_uuids, f"transport: {e!r}")
            return len(rows)

        if 200 <= resp.status_code < 300:
            self._mark_sent(event_uuids)
            return len(rows)

        # 4xx/5xx — record and try again next tick.
        # Truncate the body so a verbose error response doesn't bloat the DB.
        body_preview = (resp.text or "")[:500]
        self._mark_failed(event_uuids, f"http {resp.status_code}: {body_preview}")
        return len(rows)

    def _connect(self) -> sqlite3.Connection:
        # Each call opens a short-lived connection in WAL mode — safe to run
        # alongside the FastAPI app's connection. We don't share a connection
        # with kiosk_server because sqlite3.Connection objects are not thread-safe
        # and the worker runs in the asyncio loop while writes happen in handlers.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _select_batch(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT event_uuid, kind, payload_json "
                "FROM outbox "
                "WHERE sent_at IS NULL AND attempts < ? "
                "ORDER BY created_at "
                "LIMIT ?",
                (MAX_ATTEMPTS, self.batch_size),
            ).fetchall())

    def _mark_sent(self, event_uuids: List[str]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(event_uuids))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE outbox SET sent_at = ? WHERE event_uuid IN ({placeholders})",
                [ts, *event_uuids],
            )

    def _mark_failed(self, event_uuids: List[str], error: str) -> None:
        placeholders = ",".join("?" * len(event_uuids))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE outbox SET attempts = attempts + 1, last_error = ? "
                f"WHERE event_uuid IN ({placeholders})",
                [error, *event_uuids],
            )
            # Surface poison pills loudly — these will be ignored on future drains
            # until an operator intervenes (manual SQL fix or schema upgrade on central).
            quarantined = conn.execute(
                f"SELECT event_uuid, kind, attempts FROM outbox "
                f"WHERE event_uuid IN ({placeholders}) AND attempts >= ?",
                [*event_uuids, MAX_ATTEMPTS],
            ).fetchall()
        for row in quarantined:
            print(f"[SYNC] QUARANTINED event_uuid={row['event_uuid']} "
                  f"kind={row['kind']} attempts={row['attempts']} — manual intervention required")
