"""Bearer-auth dependency for kiosk → central API.

A FastAPI Depends(...) used by /api/sync/batch (and any future device-only
endpoints). Validates the Authorization header against the devices table,
refreshes last_seen_at, and yields a DeviceAuth(device_id, store_id) to the
handler.

Pattern note: this is a dependency rather than middleware so handlers can
declare `auth: DeviceAuth = Depends(require_device)` and get a typed value
instead of fishing off request.state. /health stays public because it doesn't
opt into the dep — middleware would have had to special-case it.
"""

import hashlib
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.sql import func

from . import db
from .models import devices


# auto_error=False so we control the 401 shape ourselves. The default behavior
# raises a 403 on a missing header, which is wrong per RFC 7235 — "no credentials"
# is 401 with a WWW-Authenticate hint, "credentials but not allowed" is 403.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class DeviceAuth:
    """Identity of the authenticated kiosk for this request."""
    device_id: str
    store_id: str


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_device(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> DeviceAuth:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("missing or non-Bearer Authorization header")

    api_key_hash = hashlib.sha256(credentials.credentials.encode("ascii")).hexdigest()

    async with db.sessionmaker()() as session:
        row = (
            await session.execute(
                select(devices.c.device_id, devices.c.store_id)
                .where(devices.c.api_key_hash == api_key_hash)
                .where(devices.c.is_active.is_(True))
            )
        ).first()
        if row is None:
            raise _unauthorized("invalid or inactive API key")

        # Refresh last_seen_at — single indexed UPDATE on the PK. Inline await
        # is cheap enough; fire-and-forget would add asyncio.create_task plumbing
        # for sub-millisecond savings.
        await session.execute(
            update(devices)
            .where(devices.c.device_id == row.device_id)
            .values(last_seen_at=func.now())
        )
        await session.commit()

    return DeviceAuth(device_id=row.device_id, store_id=row.store_id)
