# Roster Pull — End-to-End Verification Runbook

Verifies the central→kiosk roster down-channel (Step 2b slice 2): `GET /api/roster`
plus the kiosk applying what it pulls. No camera or Pi needed — runs entirely on
the Mac against the docker Postgres.

The check has **two halves**:

- **Half 1 — central HTTP side** (curl): version sequencing, incremental pull,
  biometric erasure on deactivation. This is where the watermark-collision bug
  fix is proven.
- **Half 2 — kiosk apply side** (`roster_e2e_harness.py`): the kiosk actually
  removing a face from the pkl + in-memory lists + SQLite, and the resurrection
  guard. Curl can't see this; the harness drives `RosterClient._poll_once`
  against a throwaway kiosk SQLite + pkl.

> Background: every enrollment/deactivation bumps `employees.version` from a
> store-global Postgres sequence (migration `0006`). A kiosk pulls
> `?since=<last-applied-version>`. The bug that fix closed: `version` used to be a
> per-row counter, so a store's 2nd deactivation could reuse the 1st's value and
> get silently swallowed by the scalar watermark.

---

## Prerequisites

- `conda activate face_recognition_env` (central deps live here — running the make
  targets from `(base)` is a `ModuleNotFoundError`).
- Docker Desktop running.
- `make dev` is a foreground `uvicorn --reload`, so **use two terminals**:
  Terminal 1 runs central; Terminal 2 runs the curl checks + the harness.

---

## Half 1 — central HTTP side

```bash
# ── Terminal 1: bring central up (leave running) ──
cd central
conda activate face_recognition_env
make db-up && make migrate     # idempotent: docker PG up + schema at head (0006)
make dev                       # uvicorn on 127.0.0.1:8001

# ── Terminal 2: the checks ──
conda activate face_recognition_env
curl -s http://127.0.0.1:8001/health        # {"status":"ok","db":"reachable"}

# Register a device bound to the test store. Prints the key ONCE.
# If "device already exists", pick a new DEVICE name (or delete it — see Notes).
cd central
make register-device DEVICE=mac-roster-pi STORE=mac-test-store
KEY=<paste the printed api_key>

# Pull the whole store roster. Note the watermark + which rows are is_active=true.
curl -s -H "Authorization: Bearer $KEY" "http://127.0.0.1:8001/api/roster?since=0" \
  | python -m json.tool | grep -E '"id"|"is_active"|"version"|watermark'
```

**The collision test.** Pick an employee that is currently **active** and the
watermark you just saw, then deactivate it and re-pull incrementally:

```bash
EID=<an active employee id>     # must be active; enroll one at the kiosk if none
SINCE=<watermark from the pull above>
UUID=$(uuidgen); TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Deactivation event. The handler gates on updated_at <= timestamp, so TS must be "now".
curl -s -X POST http://127.0.0.1:8001/api/sync/batch \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"events\":[{\"event_uuid\":\"$UUID\",\"kind\":\"deactivation\",\"payload\":{\"event_uuid\":\"$UUID\",\"store_id\":\"mac-test-store\",\"device_id\":\"mac-roster-pi\",\"timestamp\":\"$TS\",\"employee_id\":\"$EID\"}}]}"
# -> {"processed":1,"skipped":[]}

# Incremental re-pull.
curl -s -H "Authorization: Bearer $KEY" "http://127.0.0.1:8001/api/roster?since=$SINCE" \
  | python -m json.tool
```

**PASS looks like:** the re-pull returns **only `$EID`**, at a version **greater
than `$SINCE`** (a fresh sequence value, *not* `$SINCE + 1`), with
`is_active: false` and `encoding_b64` / `photo_b64` both `null` (biometric erased,
migration `0004`). A second deactivation in the same store lands at a yet-higher
version — never colliding with the first.

---

## Half 2 — kiosk apply side

With central running and `$KEY` exported, from the **repo root**:

```bash
export CENTRAL_API_KEY=$KEY
python roster_e2e_harness.py --store mac-test-store
```

The harness is self-discovering — it pulls the live roster and picks rows out of
it, seeding a throwaway kiosk SQLite + pkl for each test (never touches your real
`data/kiosk.db` or `data/*.pkl`):

- **Test A — deactivation apply:** takes an **inactive** central row, seeds a
  kiosk where that person is still active, polls once, and asserts the face is
  gone from the pkl + in-memory lists + SQLite (soft-delete) and the watermark
  advanced.
- **Test B — resurrection guard:** takes an **active** central row, seeds a kiosk
  with an unsent local `deactivation` for that person, polls, and asserts the face
  is **held** (not re-added) and the watermark is not advanced past the held row.

A test prints `SKIP` (not `FAIL`) if the live roster lacks a row of the kind it
needs. **To exercise both, the store needs at least one active and one inactive
employee.** Expected clean result:

```
Test A: PASS ✓  (applied deactivation of <id> (v<n>))
Test B: PASS ✓  (held active <id> (v<n>) behind unsent local delete)
```

Exit code is non-zero if any test `FAIL`s.

---

## Notes / gotchas

- **Stale data is the #1 confuser.** `mac-test-store`'s employees and versions
  change every time you run this. Don't hardcode expected versions — read the
  watermark from the pull and act on whatever is currently active/inactive.
- **`make register-device` prints the key once** (central stores only the sha256).
  To reuse a `DEVICE` name, delete it first:
  `docker exec -it fr-central-postgres psql -U central -d central -c "DELETE FROM devices WHERE device_id='mac-roster-pi';"`
- **`conda run -n env python -` and heredocs swallow piped stdin.** In an
  interactive shell (`conda activate`) this is a non-issue; only relevant if you
  script it through `conda run` — write the script to a file and run it by path.
- **Port 8001 in use** (`Errno 48` on `make dev`): `kill $(lsof -ti:8001)`.
- **Inspect central directly** (API not required):
  `docker exec fr-central-postgres psql -U central -d central -c "SELECT id,is_active,version,(encoding IS NOT NULL) AS has_enc FROM employees WHERE store_id='mac-test-store' ORDER BY version;"`
- This verifies **correctness** (auth, scoping, sequencing, apply, guard) — all
  hardware-independent. Pi-specific concerns (camera, ONNX-on-ARM, NTP, thermals)
  are a separate milestone (`PI_SETUP.md`).
```
