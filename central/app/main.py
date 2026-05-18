"""Central tier FastAPI app — entry point.

Step 2a skeleton: only exposes /health. Auth middleware, /api/sync/batch, and
admin endpoints land in follow-up commits.

Run locally:
    cd central && make dev
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from . import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Touch the engine so misconfigured DATABASE_URL fails fast at startup
    # rather than on the first request.
    db.engine()
    yield
    await db.dispose()


app = FastAPI(title="FR Central", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    """Readiness check — pings Postgres. 200 only if the DB round-trips."""
    try:
        async with db.engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "db": "unreachable", "error": repr(e)},
        )
    return JSONResponse(content={"status": "ok", "db": "reachable"})
