#!/usr/bin/env python3
"""
Roster client — pulls central's per-store roster DOWN into this kiosk.

The mirror image of sync_worker.py. Where the sync worker drains the local
outbox UP to central, this polls `GET /api/roster?since=<version>` and applies
what central reports for THIS store: primarily HQ-initiated deactivations once
the Step 2b admin UI lands, plus this kiosk's own enrollments echoed back once
(harmless — the apply is idempotent).

Runs as a background asyncio task inside kiosk_server.py's FastAPI lifespan, on
the same event loop as the recognition endpoints. That co-location is load-
bearing: coroutines on one loop interleave only at `await` points, so this task
can reassign app.state.known_encodings/known_labels in a single statement (no
`await` between read and assign) and a concurrent /api/recognize call sees
either the old lists or the new ones, never a torn state — the exact pattern the
enroll/delete endpoints already rely on.

Incremental via a watermark. Each enrollment/deactivation bumps central's
employees.version; we persist the highest version applied in the local
`sync_state` table and pull `?since=<that>` so each poll fetches only changes.

Idempotent + self-correcting:
  - An is_active row upserts the face (pkl + in-memory + employees row + photo).
  - An is_active=false row removes it (pkl + in-memory + soft-delete + photo),
    WITHOUT re-queuing a deactivation outbox event — the change originated
    upstream, so echoing it back up would be a loop.

Resurrection guard: the kiosk is both a writer (local delete -> queued
deactivation, not yet synced up) and a reader (this pull). If a poll fires
before a local deletion reaches central, central still reports that employee as
active. Applying that blindly would re-add a just-fired employee's face. So
before applying an is_active row we check the outbox for an unsent deactivation
for that employee; if present, local intent wins — we skip the row and hold the
watermark so it is re-evaluated next poll (by which time the deactivation has
synced up and central agrees).

Fully passive when --central-url is unset: kiosk_server.py simply doesn't start
it (same as the sync worker).
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

import pkl_store


# Watermark key in the local `sync_state` table.
_WATERMARK_KEY = "roster_version"


class RosterClient:
    """Background poll loop that applies central's roster to local state."""

    def __init__(
        self,
        app,
        central_url: str,
        api_key: str,
        pkl_path: str,
        store_id: str,
        expected_embedder: str,
        interval_seconds: int = 1800,
        limit: int = 200,
        request_timeout_seconds: float = 30.0,
    ):
        self.app = app
        self.roster_url = central_url.rstrip("/") + "/api/roster"
        self.api_key = api_key
        self.pkl_path = pkl_path
        self.store_id = store_id
        self.expected_embedder = expected_embedder
        self.interval_seconds = interval_seconds
        self.limit = limit
        self.request_timeout_seconds = request_timeout_seconds
        # Enrollment photos live next to the pkl, in data/employees/<id>.jpg —
        # same convention as the enroll endpoint.
        self.employees_dir = os.path.join(os.path.dirname(pkl_path) or ".", "employees")
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # ── lifecycle (mirrors SyncWorker) ──────────────────────────────────────

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="roster_client")

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
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            print(f"[ROSTER] client started — target={self.roster_url}, "
                  f"store={self.store_id}, interval={self.interval_seconds}s")
            while not self._stop_event.is_set():
                try:
                    applied, more = await self._poll_once(client)
                    # If central returned a full page AND we applied all of it
                    # (no rows held back by the resurrection guard), there may be
                    # more behind it — pull again immediately instead of waiting
                    # a full interval. `more` is False when a row was skipped, so
                    # a stuck local deletion can't spin this into a tight loop.
                    if more:
                        continue
                except Exception as e:
                    # Never let a poll bug crash the server.
                    print(f"[ROSTER] tick failed unexpectedly: {e!r}")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    pass
            print("[ROSTER] client stopped")

    # ── one poll ────────────────────────────────────────────────────────────

    async def _poll_once(self, client: httpx.AsyncClient) -> Tuple[int, bool]:
        """Pull one page and apply it. Returns (rows_applied, pull_again_now)."""
        since = self._read_watermark()

        try:
            resp = await client.get(
                self.roster_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"since": since, "limit": self.limit},
            )
        except httpx.HTTPError as e:
            # Central unreachable — recognition keeps running on the cached pkl;
            # we just retry next interval. Same graceful-offline posture as sync.
            print(f"[ROSTER] pull failed (offline?): {e!r}")
            return 0, False

        if not (200 <= resp.status_code < 300):
            print(f"[ROSTER] pull HTTP {resp.status_code}: {(resp.text or '')[:300]}")
            return 0, False

        body = resp.json()
        rows: List[Dict[str, Any]] = body.get("roster", [])
        if not rows:
            return 0, False

        applied = 0
        # Highest version we're allowed to advance the watermark to. We start at
        # the page's max version and pull it DOWN to (skipped_version - 1) for any
        # row the resurrection guard held back, so the held row is re-fetched next
        # poll. Versions are global and ascending, so this re-fetches the held row
        # and everything after it — all idempotent on re-apply.
        page_max = max(r["version"] for r in rows)
        new_watermark = page_max
        held_back = False

        for r in rows:
            try:
                if self._apply_row(r):
                    applied += 1
            except _HoldRow:
                held_back = True
                new_watermark = min(new_watermark, r["version"] - 1)
            except Exception as e:
                # A single malformed/odd row shouldn't stall the whole roster.
                # Log it, skip it, but DON'T advance past it — hold the watermark
                # below it so a fixed central row (or a code fix) is retried.
                print(f"[ROSTER] failed to apply id={r.get('id')!r} v={r.get('version')}: {e!r}")
                held_back = True
                new_watermark = min(new_watermark, r["version"] - 1)

        self._write_watermark(new_watermark)

        # Only chase the next page immediately if we consumed a full page cleanly.
        pull_again = (len(rows) == self.limit) and not held_back
        if applied:
            print(f"[ROSTER] applied {applied} change(s); watermark {since} -> {new_watermark}")
        return applied, pull_again

    def _apply_row(self, r: Dict[str, Any]) -> bool:
        """Apply one roster entry. Returns True if a change was made.

        Raises _HoldRow to signal "skip and hold the watermark below this row".
        """
        employee_id = r["id"]

        if r.get("is_active"):
            # ── Resurrection guard ──────────────────────────────────────────
            # Local deletion intent that central hasn't acknowledged yet wins.
            if self._has_unsent_deactivation(employee_id):
                print(f"[ROSTER] holding active row for {employee_id!r} — local "
                      f"deactivation still unsent (resurrection guard)")
                raise _HoldRow()
            return self._apply_active(r)

        return self._apply_inactive(employee_id)

    # ── apply: active (enroll/update) ────────────────────────────────────────

    def _apply_active(self, r: Dict[str, Any]) -> bool:
        employee_id = r["id"]
        conn = self.app.state.db_conn

        # In the single-Pi-per-store topology the only active rows we ever see
        # are employees THIS kiosk enrolled (echoed back) — so a local row
        # already exists and this is an UPDATE. A brand-new active employee with
        # no local row is the replacement-Pi / full-rebuild path, which is out of
        # scope for slice 2 (it also can't satisfy the local pos_employee_id NOT
        # NULL trigger, since the roster payload omits pos_id by design). Skip it
        # with a warning rather than fail; advance past it (it's a deferred
        # feature, not a transient conflict, so holding the watermark would just
        # re-warn every poll).
        local = conn.execute(
            "SELECT id FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
        if local is None:
            print(f"[ROSTER] active employee {employee_id!r} not enrolled locally — "
                  f"skipping (full-rebuild path not implemented in slice 2)")
            return False

        # Embedder must match — a 512-D ArcFace vector can't live alongside a
        # 128-D dlib one (distance math would crash). Mismatch = skip-and-warn.
        embedder_type = r.get("embedder_type")
        if embedder_type != self.expected_embedder:
            print(f"[ROSTER] embedder mismatch for {employee_id!r} "
                  f"(roster={embedder_type}, local={self.expected_embedder}) — skipping")
            return False

        encoding_b64 = r.get("encoding_b64")
        if not encoding_b64:
            # Active but no biometric — shouldn't happen (active rows carry an
            # encoding on central). Nothing to load into the pkl, so skip.
            print(f"[ROSTER] active employee {employee_id!r} has no encoding — skipping")
            return False

        embedding = np.frombuffer(base64.b64decode(encoding_b64), dtype=np.float32)

        # 1) pkl on disk (atomic) — recognition substrate.
        pkl_store.upsert(self.pkl_path, employee_id, embedding, source="roster_sync")

        # 2) in-memory lists — single atomic reassignment, no await in between,
        #    so a concurrent /api/recognize sees old-or-new, never half.
        self._memory_upsert(employee_id, embedding)

        # 3) local employees row — keep display_name / store_id / active in step.
        with conn:
            conn.execute(
                "UPDATE employees SET is_active = 1, name = ?, store_id = ? WHERE id = ?",
                (r.get("display_name") or employee_id, r.get("store_id") or self.store_id, employee_id),
            )

        # 4) photo (best-effort — non-fatal; recognition doesn't need it).
        photo_b64 = r.get("photo_b64")
        if photo_b64:
            try:
                os.makedirs(self.employees_dir, exist_ok=True)
                tmp = os.path.join(self.employees_dir, f".{employee_id}.jpg.tmp")
                final = os.path.join(self.employees_dir, f"{employee_id}.jpg")
                with open(tmp, "wb") as f:
                    f.write(base64.b64decode(photo_b64))
                os.replace(tmp, final)
            except Exception as e:
                print(f"[ROSTER] WARN: could not write photo for {employee_id!r}: {e}")

        print(f"[ROSTER] upserted {employee_id!r} from central")
        return True

    # ── apply: inactive (deactivation from upstream) ─────────────────────────

    def _apply_inactive(self, employee_id: str) -> bool:
        conn = self.app.state.db_conn

        row = conn.execute(
            "SELECT photo_path, is_active FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
        # Unknown locally, or already inactive — nothing to do. (Idempotent: a
        # re-pull of the same deactivation lands here as a no-op.)
        if row is None or row[1] == 0:
            return False
        photo_path = row[0]

        # 1) pkl wipe (atomic) — stop matching this face across restarts.
        pkl_store.remove(self.pkl_path, employee_id)

        # 2) in-memory removal — atomic reassignment, mirrors delete_employee.
        st = self.app.state
        pairs = [
            (enc, lbl)
            for enc, lbl in zip(st.known_encodings, st.known_labels)
            if lbl != employee_id
        ]
        if pairs:
            st.known_encodings, st.known_labels = map(list, zip(*pairs))
        else:
            st.known_encodings, st.known_labels = [], []

        # 3) soft-delete the local row. NO outbox event — this deactivation came
        #    FROM central; re-queuing it would echo it straight back up.
        with conn:
            conn.execute(
                "UPDATE employees SET is_active = 0 WHERE id = ? AND is_active = 1",
                (employee_id,),
            )

        # 4) photo + transient state cleanup (best-effort).
        if photo_path:
            try:
                os.remove(photo_path)
            except OSError:
                pass
        st.cooldown.pop(employee_id, None)
        st.pending.pop(employee_id, None)

        print(f"[ROSTER] deactivated {employee_id!r} per central")
        return True

    # ── outbox check (resurrection guard) ────────────────────────────────────

    def _has_unsent_deactivation(self, employee_id: str) -> bool:
        """True if an undelivered local `deactivation` exists for this employee.

        Such a row means the manager deleted this person at the kiosk and the
        intent hasn't reached central yet, so central's "active" view is stale.
        We parse the JSON payload rather than LIKE-matching to avoid false hits
        on a substring of another field.
        """
        conn = self.app.state.db_conn
        rows = conn.execute(
            "SELECT payload_json FROM outbox "
            "WHERE kind = 'deactivation' AND sent_at IS NULL"
        ).fetchall()
        for (payload_json,) in rows:
            try:
                if json.loads(payload_json).get("employee_id") == employee_id:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    def _memory_upsert(self, label: str, embedding: np.ndarray) -> None:
        """Atomically replace `label`'s in-memory encoding (drop-then-append)."""
        st = self.app.state
        new_enc, new_lbl = [], []
        for enc, lbl in zip(st.known_encodings, st.known_labels):
            if lbl != label:
                new_enc.append(enc)
                new_lbl.append(lbl)
        new_enc.append(embedding.tolist())
        new_lbl.append(label)
        # Single reassignment — no await between here and the reads in recognize.
        st.known_encodings, st.known_labels = new_enc, new_lbl

    # ── watermark persistence ────────────────────────────────────────────────

    def _read_watermark(self) -> int:
        row = self.app.state.db_conn.execute(
            "SELECT value FROM sync_state WHERE key = ?", (_WATERMARK_KEY,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (ValueError, TypeError):
            return 0

    def _write_watermark(self, version: int) -> None:
        conn = self.app.state.db_conn
        with conn:
            conn.execute(
                "INSERT INTO sync_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_WATERMARK_KEY, str(version)),
            )


class _HoldRow(Exception):
    """Internal signal: skip this roster row and hold the watermark below it."""
