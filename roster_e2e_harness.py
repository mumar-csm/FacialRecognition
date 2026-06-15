#!/usr/bin/env python3
"""Roster pull — Half 2 (kiosk apply) end-to-end harness.

Half 1 (the central HTTP side: version sequencing, incremental pull, biometric
erasure) is a curl exercise — see central/ROSTER_E2E.md. This script covers the
half curl can't: the kiosk actually APPLYING what GET /api/roster reports, by
driving roster_client.RosterClient._poll_once against a throwaway kiosk
SQLite + pkl. Nothing here touches your real data/kiosk.db or data/*.pkl.

It is self-discovering: it pulls the live roster and picks rows out of it, so it
adapts to whatever state mac-test-store is in.

  Test A — deactivation apply: takes an INACTIVE central row, seeds a kiosk that
    still has that person active, polls, and asserts the face is removed from the
    pkl + in-memory lists + SQLite (soft-delete) and the watermark advances.

  Test B — resurrection guard: takes an ACTIVE central row, seeds a kiosk that has
    an unsent local `deactivation` for that person, polls, and asserts the face is
    HELD (not re-added) and the watermark is not advanced past the held row.

A test SKIPs (not fails) if the live roster has no row of the kind it needs — to
exercise both, have one active and one inactive employee in the store.

Usage (from the repo root, in face_recognition_env, with central running):
    export CENTRAL_API_KEY=<key from `make register-device`>
    python roster_e2e_harness.py --store mac-test-store
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import types

import httpx
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import pkl_store  # noqa: E402
from build_encodings import EncodingsDB  # noqa: E402
from roster_client import RosterClient  # noqa: E402

_KIOSK_DDL = """
CREATE TABLE employees (id TEXT PRIMARY KEY, name TEXT NOT NULL, enrolled_at TEXT NOT NULL,
  photo_path TEXT, is_active INTEGER NOT NULL DEFAULT 1, store_id TEXT NOT NULL DEFAULT 'store-01',
  pos_employee_id TEXT);
CREATE TABLE outbox (event_uuid TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL, sent_at TEXT, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT);
CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _seed_kiosk(emp_id, store, embedder, watermark, with_unsent_deactivation=False):
    """Build a throwaway kiosk (sqlite + pkl + fake app) with emp_id active."""
    work = tempfile.mkdtemp(prefix="roster_e2e_")
    db_path = os.path.join(work, "kiosk.db")
    pkl_path = os.path.join(work, "faces.pkl")

    conn = sqlite3.connect(db_path)
    conn.executescript(_KIOSK_DDL)
    conn.execute(
        "INSERT INTO employees (id,name,enrolled_at,is_active,store_id,pos_employee_id) "
        "VALUES (?,?,?,1,?,?)", (emp_id, emp_id, "2026-01-01T00:00:00Z", store, "1234567"))
    conn.execute("INSERT INTO sync_state (key,value) VALUES ('roster_version',?)", (str(watermark),))
    if with_unsent_deactivation:
        conn.execute(
            "INSERT INTO outbox (event_uuid,kind,payload_json,created_at,sent_at) VALUES (?,?,?,?,NULL)",
            ("local-deact", "deactivation",
             json.dumps({"employee_id": emp_id, "store_id": store}), "2026-01-01T00:00:00Z"))
    conn.commit()

    emb = np.random.rand(512 if embedder == "arcface" else 128).astype(np.float32)
    emb /= np.linalg.norm(emb)
    pkl_store.save(pkl_path, EncodingsDB(
        encodings=[emb.tolist()], labels=[emp_id], meta=[{"label": emp_id}],
        embedding_dim=emb.shape[0], embedder_type=embedder))

    app = types.SimpleNamespace(state=types.SimpleNamespace(
        db_conn=conn, known_encodings=[emb.tolist()], known_labels=[emp_id],
        cooldown={emp_id: 0.0}, pending={emp_id: "x"}))
    return app, conn, pkl_path


def _watermark(conn):
    return int(conn.execute("SELECT value FROM sync_state WHERE key='roster_version'").fetchone()[0])


def _in_pkl(pkl_path, emp_id):
    return emp_id in pkl_store.load(pkl_path).labels


async def _run(args):
    key = args.key or os.environ.get("CENTRAL_API_KEY")
    if not key:
        sys.exit("No API key — pass --key or set CENTRAL_API_KEY (from `make register-device`).")
    roster_url = args.central.rstrip("/") + "/api/roster"

    # Discover live roster state.
    with httpx.Client(timeout=15) as c:
        r = c.get(roster_url, headers={"Authorization": f"Bearer {key}"}, params={"since": 0})
    if r.status_code != 200:
        sys.exit(f"roster pull failed: HTTP {r.status_code}: {r.text[:300]}")
    rows = r.json()["roster"]
    print(f"discovered {len(rows)} row(s) in {args.store}: " +
          ", ".join(f"{x['id']}(v{x['version']},{'active' if x['is_active'] else 'inactive'})" for x in rows))

    inactive = next((x for x in rows if not x["is_active"]), None)
    active = next((x for x in rows if x["is_active"]), None)
    results = {}

    # ── Test A — deactivation apply ──────────────────────────────────────────
    if inactive is None:
        results["A"] = ("SKIP", "no inactive central row to apply")
    else:
        eid, ver = inactive["id"], inactive["version"]
        app, conn, pkl_path = _seed_kiosk(eid, args.store, args.embedder, watermark=ver - 1)
        client = RosterClient(app, args.central, key, pkl_path, args.store, expected_embedder=args.embedder)
        async with httpx.AsyncClient(timeout=15) as c:
            await client._poll_once(c)
        removed = (not _in_pkl(pkl_path, eid)
                   and eid not in app.state.known_labels
                   and conn.execute("SELECT is_active FROM employees WHERE id=?", (eid,)).fetchone()[0] == 0
                   and _watermark(conn) >= ver)
        results["A"] = ("PASS" if removed else "FAIL", f"applied deactivation of {eid} (v{ver})")

    # ── Test B — resurrection guard ──────────────────────────────────────────
    if active is None:
        results["B"] = ("SKIP", "no active central row to guard against")
    else:
        eid, ver = active["id"], active["version"]
        app, conn, pkl_path = _seed_kiosk(eid, args.store, args.embedder, watermark=ver - 1,
                                          with_unsent_deactivation=True)
        client = RosterClient(app, args.central, key, pkl_path, args.store, expected_embedder=args.embedder)
        async with httpx.AsyncClient(timeout=15) as c:
            await client._poll_once(c)
        held = (_in_pkl(pkl_path, eid)
                and eid in app.state.known_labels
                and _watermark(conn) < ver)
        results["B"] = ("PASS" if held else "FAIL", f"held active {eid} (v{ver}) behind unsent local delete")

    print()
    for t, (status, note) in sorted(results.items()):
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "—"}[status]
        print(f"Test {t}: {status} {mark}  ({note})")
    if any(s == "FAIL" for s, _ in results.values()):
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Roster pull Half 2 (kiosk apply) harness")
    p.add_argument("--central", default="http://127.0.0.1:8001")
    p.add_argument("--store", default="mac-test-store")
    p.add_argument("--embedder", default="arcface", choices=["arcface", "dlib"])
    p.add_argument("--key", default=None, help="device API key (else $CENTRAL_API_KEY)")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
