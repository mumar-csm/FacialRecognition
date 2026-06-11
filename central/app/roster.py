"""GET /api/roster — incremental, store-scoped roster pull for kiosks.

A kiosk polls this on a slow cadence (default 1h) to learn about roster changes
it did not originate — primarily HQ-initiated deactivations once the Step 2b
admin UI lands. In the single-Pi-per-store topology the kiosk is the only
enroller for its store, so steady-state pulls mostly carry deactivations; the
kiosk's own enrollments echo back once (harmless, idempotent) the first time
their version clears the watermark.

Two invariants:

1. **Store scoping is derived from the device, never the client.** The store
   comes off DeviceAuth (the authenticated API key), not a query param. A
   misconfigured or stolen Pi therefore cannot pull a different store's roster.
   Mirrors how sync.py rejects any event whose payload.store_id != auth.store_id.

2. **Incremental by `version`.** Every enrollment/deactivation bumps
   employees.version (see sync.py). `?since=<last-seen-version>` returns only
   rows that changed, ordered by version ascending and capped by `limit`, so a
   kiosk that is far behind catches up across several polls, advancing its
   watermark each time.

pos_employee_id is deliberately NOT in the payload: locked decision #8 keeps
pos_id flowing edge->central only. Pushing it down is the unresolved edit-flow
(Option 2) and is out of scope here. The kiosk keeps its own copy.
"""

import base64
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from . import db
from .auth import DeviceAuth, require_device
from .models import employees


router = APIRouter()


# Cap rows per response. A kiosk further behind re-pulls on its next tick (it
# advances `since` to the max version it received). Bounds payload size — photos
# make each active row non-trivial.
_DEFAULT_LIMIT = 200


@router.get("/api/roster")
async def get_roster(
    since: int = Query(0, ge=0),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=1000),
    auth: DeviceAuth = Depends(require_device),
) -> dict[str, Any]:
    async with db.sessionmaker()() as session:
        rows = (
            await session.execute(
                select(
                    employees.c.id,
                    employees.c.store_id,
                    employees.c.display_name,
                    employees.c.is_active,
                    employees.c.embedder_type,
                    employees.c.embedding_dim,
                    employees.c.encoding,
                    employees.c.photo,
                    employees.c.version,
                )
                .where(employees.c.store_id == auth.store_id)
                .where(employees.c.version > since)
                .order_by(employees.c.version.asc())
                .limit(limit)
            )
        ).all()

    roster: list[dict[str, Any]] = []
    for r in rows:
        roster.append(
            {
                "id": r.id,
                "store_id": r.store_id,
                "display_name": r.display_name,
                "is_active": r.is_active,
                "embedder_type": r.embedder_type,
                "embedding_dim": r.embedding_dim,
                # encoding/photo are NULL for deactivated rows — the deactivation
                # handler erases the biometric (migration 0004). Active rows always
                # carry an encoding; photo may be NULL for older rows.
                "encoding_b64": base64.b64encode(r.encoding).decode("ascii")
                if r.encoding is not None
                else None,
                "photo_b64": base64.b64encode(r.photo).decode("ascii")
                if r.photo is not None
                else None,
                "version": r.version,
            }
        )

    return {
        "roster": roster,
        # The max version in this page. The kiosk can compute it itself, but
        # returning it avoids an empty-page edge case on its side and documents
        # the contract: "you are now caught up to here (for the rows you applied)".
        "watermark": roster[-1]["version"] if roster else since,
    }
