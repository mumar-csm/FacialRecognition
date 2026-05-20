# FR Central

Cloud-side companion to the kiosk. Receives sync uploads from every Pi,
serves the roster pull, and (eventually) the HQ admin UI.

This is the **Step 2a skeleton** — only `/health` is wired. Auth middleware,
`/api/sync/batch`, the device-registration CLI, and the admin UI land in
follow-up commits.

## Prerequisites

- Docker Desktop running (for the local Postgres container)
- The `face_recognition_env` conda env (same one the kiosk uses)

## Local dev loop

```bash
cd central
cp .env.example .env                 # one-time
conda activate face_recognition_env
make install                         # one-time: pip install requirements
make db-up                           # start Postgres in Docker
make migrate                         # apply Alembic migrations
make dev                             # uvicorn on http://127.0.0.1:8001
```

Verify it's alive:

```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","db":"reachable"}
```

If you see `{"status":"unhealthy", ...}`, Postgres isn't reachable —
`make db-logs` to investigate, or `make db-reset` to nuke and recreate.

## Useful targets

| target             | what it does                                              |
|--------------------|-----------------------------------------------------------|
| `make db-up`       | start Postgres container (idempotent)                     |
| `make db-down`     | stop container, **preserve** data                         |
| `make db-reset`    | **destroy** the data volume + recreate                    |
| `make migrate`     | apply pending Alembic migrations                          |
| `make revision MSG='...'` | create a new empty migration script               |
| `make psql`        | drop into psql against the local DB                       |
| `make dev`         | run uvicorn with autoreload                               |

## Schema

Initial tables (`alembic/versions/0001_init.py`):
- `devices` — per-Pi API key store (sha256 hash of the bearer token)
- `employees` — cross-store roster; multi-tenant by `store_id`
- `attendance` — idempotent on `event_uuid`
- `spoof_attempts` — idempotent on `event_uuid`

`pos_id` tables and the `users` / `admin_audit_log` tables land in later
migrations; see the plan file for the breakdown.

## Layout

```
central/
├── app/
│   ├── main.py        # FastAPI entry, /health
│   ├── db.py          # async engine + sessionmaker
│   └── models.py      # SQLAlchemy Core table defs
├── alembic/
│   ├── env.py         # async-aware migration runner
│   ├── script.py.mako
│   └── versions/
│       └── 0001_init.py
├── alembic.ini
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
```
