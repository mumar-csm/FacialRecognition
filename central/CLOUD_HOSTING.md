# Cloud-Hosting the Central Tier — Design

**Status:** design / decision doc. No deploy artifacts (Dockerfile, prod unit)
exist yet — this captures the recommended shape so the next session can build it.

## Problem

Central today only ever runs locally: `docker-compose.yml` brings up Postgres
bound to `127.0.0.1:5432`, and `make dev` runs `uvicorn --reload` on
`127.0.0.1:8001`. Nothing is reachable by a remote Pi. For a real multi-store
pilot, the fleet needs a stable, always-on, reachable central endpoint — without
turning central into an internet-exposed attack surface.

## Decision: tailnet-only, single small VM

Run central on **one small always-on VM joined to the existing Tailscale
tailnet**. No public ports. Pis (already on the tailnet per `PI_SETUP.md`) reach
it by its MagicDNS name. This was chosen over a public PaaS and over a
self-managed public VPS because:

- The fleet is already on Tailscale, so reusing it adds **zero new attack
  surface** — central has no public listener at all.
- Tailscale device auth becomes a network-layer second factor *on top of* the
  per-device bearer key. This meaningfully defers the "move to mTLS before >10
  stores" trigger in [[project-central-tier-topology]].
- Provider-agnostic: reachability is via the tailnet, so the VM can live on
  Hetzner / DigitalOcean / a Fly machine / EC2 `t4g.small` — wherever is cheapest.

```
  Pi (tailnet) ──wireguard──> VM: tailscale serve (TLS) ──> uvicorn :8001
                                          │
                                          └──> Postgres (docker, 127.0.0.1:5432)
   no public ports open; VM firewall allows only the tailscale0 interface
```

## Components

### 1. Reachability / TLS — `tailscale serve`

Expose the app as `https://fr-central.<tailnet>.ts.net` via:

```bash
sudo tailscale serve --bg 8001
```

Tailscale provisions a real Let's Encrypt cert for the `*.ts.net` name
automatically (no nginx, no cert renewal cron). This keeps the kiosk's `https://`
`central_url` assumption valid and gives defense-in-depth even though wireguard
already encrypts the hop.

- **`central_url` to put in each Pi's `device.json`:** `https://fr-central.<tailnet>.ts.net`
- **Acceptable fallback:** plain `http://fr-central:8001` over the tailnet (MagicDNS
  name, no `tailscale serve`). Drop TLS, rely on wireguard. Simpler, but the kiosk
  example URLs assume https — prefer `serve`.

### 2. App process — `fr-central.service` (production ASGI)

Replace `make dev` (which has `--reload`, single worker, dev-only) with a systemd
unit running uvicorn without reload and with a few workers:

```ini
[Service]
EnvironmentFile=/etc/fr-central/central.env      # DATABASE_URL (uncommitted)
WorkingDirectory=/opt/fr-central/central
ExecStart=/opt/fr-central/venv/bin/uvicorn app.main:app \
    --host 127.0.0.1 --port 8001 --workers 3
Restart=on-failure
```

- Bind `127.0.0.1` — `tailscale serve` proxies to it; nothing else should reach it.
- Each worker gets its own async engine + pool via the existing `app/main.py`
  lifespan and `app/db.py` — multi-worker is safe as-is.
- `--workers N`: start with 3 on a 2-vCPU box; the workload is light (batch sync
  every 30 min/store, roster pulls), so this is headroom, not a bottleneck.

### 3. Database — Postgres on the same VM (for now)

Keep the existing `central/docker-compose.yml` Postgres **on the VM**. Rationale:

- It's tailnet-native — no public endpoint, no extra plumbing. A managed PG
  (RDS/Render/Supabase) exposes a public or VPC endpoint that the tailnet can't
  reach without a subnet router; not worth that complexity at pilot scale.
- The named volume `central_pgdata` already persists across restarts.

**This makes backups your responsibility** — add them before storing real data:

- Nightly `pg_dump` (cron or a `fr-central-backup.timer`) →
  `docker exec fr-central-postgres pg_dump -U central central | gzip > …` →
  push to S3-compatible object storage (Backblaze B2 / S3 / R2).
- **Document and actually run a restore drill** — an untested backup isn't one.
- Revisit managed Postgres (for HA + point-in-time recovery) when the fleet or
  the compliance bar grows. The `DATABASE_URL`-only coupling makes that swap easy.

### 4. Secrets / config

- `DATABASE_URL` lives in `/etc/fr-central/central.env` (systemd `EnvironmentFile`,
  mode 600, uncommitted) — mirrors the kiosk's `secrets.env` pattern. For the
  on-box docker Postgres it stays
  `postgresql+asyncpg://central:central@localhost:5432/central`; change the
  password from the compose default before real data lands.
- `central/.env` / `.env.example` remain the **local-dev** path only.

### 5. Schema migrations on deploy

`alembic upgrade head` (i.e. `make migrate`) as an explicit deploy step, run
after pulling new code and before restarting `fr-central.service`. Migrations
stay hand-written per the locked decision in [[project-step-2a-status]].

## Cutover steps (next-session checklist)

1. Provision the VM; install Docker + Tailscale; `tailscale up` (tag it, e.g.
   `tag:central`).
2. Clone repo to `/opt/fr-central`; create venv; `make install`.
3. Write `/etc/fr-central/central.env` with `DATABASE_URL` (+ a non-default PG
   password); update `docker-compose.yml` password to match.
4. `make db-up && make migrate`.
5. Install + enable `fr-central.service`; `curl 127.0.0.1:8001/health` →
   `{"status":"ok","db":"reachable"}`.
6. `sudo tailscale serve --bg 8001`; from a Pi, `curl https://fr-central.<tailnet>.ts.net/health`.
7. `make register-device …` per device; set each Pi's `device.json` `central_url`
   to the serve URL (via `provision_device.py`).
8. Add the nightly `pg_dump` backup job; run one restore drill.

## Rough cost

- VM: ~$5–12/mo (1–2 vCPU, 1–2 GB) — fine for a small fleet's batch workload.
- Tailscale: free tier covers a pilot (100 devices / 3 users).
- Object storage for backups: cents/mo at this volume.

## Open / deferred

- **mTLS** (the topology doc's >10-store trigger) — tailnet identity softens the
  urgency; revisit at franchise scale.
- **Managed Postgres + HA** — deferred until uptime/compliance demands it.
- **Multiple central instances / load balancing** — not needed; one VM with a few
  workers is ample for batch-cadence traffic.
