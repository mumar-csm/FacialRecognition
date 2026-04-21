#!/usr/bin/env python3
"""
Sync SQLite employees table with the face encodings pkl file.

Useful whenever the two stores drift out of sync — e.g. after CLI enrollment
via build_encodings.py, manual pkl edits, or database migration.

Safe to run multiple times: uses INSERT OR IGNORE so existing records are
never overwritten or duplicated.

Usage:
    python backfill_employees.py
    python backfill_employees.py --database data/known_faces_arcface.pkl --sqlite data/kiosk.db
    python backfill_employees.py --dry-run
"""

import argparse
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from build_encodings import EncodingsDB  # noqa: F401 — required for pickle deserialization


def parse_args():
    p = argparse.ArgumentParser(description="Sync employees table with pkl embeddings.")
    p.add_argument("--database", default="data/known_faces_arcface.pkl",
                   help="Path to face encodings pkl (default: data/known_faces_arcface.pkl)")
    p.add_argument("--sqlite", default="data/kiosk.db",
                   help="Path to kiosk SQLite database (default: data/kiosk.db)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview changes without writing anything")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.database):
        print(f"ERROR: pkl not found: {args.database}")
        sys.exit(1)
    if not os.path.exists(args.sqlite):
        print(f"ERROR: SQLite db not found: {args.sqlite}")
        sys.exit(1)

    # ── Load pkl labels ──
    db = pickle.load(open(args.database, "rb"))
    pkl_labels = set(db.labels)
    print(f"pkl:    {len(pkl_labels)} unique employees")

    # ── Load SQLite labels ──
    conn = sqlite3.connect(args.sqlite)
    rows = conn.execute("SELECT id, is_active FROM employees").fetchall()
    sqlite_all    = {r[0] for r in rows}
    sqlite_active = {r[0] for r in rows if r[1] == 1}
    print(f"SQLite: {len(sqlite_active)} active, {len(sqlite_all - sqlite_active)} inactive")

    # ── Diff ──
    missing_in_sqlite  = pkl_labels - sqlite_all
    inactive_in_sqlite = pkl_labels & (sqlite_all - sqlite_active)
    orphans_in_sqlite  = sqlite_active - pkl_labels

    print()

    if missing_in_sqlite:
        print(f"To INSERT ({len(missing_in_sqlite)} in pkl but missing from SQLite):")
        for label in sorted(missing_in_sqlite):
            print(f"  + {label}")
    else:
        print("No missing employees to insert.")

    if inactive_in_sqlite:
        print(f"\nSkipped ({len(inactive_in_sqlite)} in pkl but soft-deleted in SQLite — reactivate manually if intended):")
        for label in sorted(inactive_in_sqlite):
            print(f"  ~ {label} (inactive)")

    if orphans_in_sqlite:
        print(f"\nOrphans ({len(orphans_in_sqlite)} active in SQLite but not in pkl — may need re-enrollment):")
        for label in sorted(orphans_in_sqlite):
            print(f"  ? {label}")

    if not missing_in_sqlite:
        print("Nothing to do — stores are in sync.")
        conn.close()
        return

    if args.dry_run:
        print("\nDry run — no changes written.")
        conn.close()
        return

    # ── Backfill ──
    now = datetime.now(timezone.utc).isoformat()
    backfilled = 0
    for label in sorted(missing_in_sqlite):
        name = label.replace("_", " ").title()
        conn.execute(
            "INSERT OR IGNORE INTO employees (id, name, enrolled_at, is_active) VALUES (?, ?, ?, 1)",
            (label, name, now),
        )
        backfilled += 1

    conn.commit()
    conn.close()
    print(f"\nDone — {backfilled} employee(s) inserted.")


if __name__ == "__main__":
    main()
