#!/usr/bin/env python3
"""
Atomic read-modify-write helpers for the recognition pkl.

Shared by kiosk_server.py (enroll/delete/reconcile) and roster_client.py
(central down-sync). Lives in its own module — NOT kiosk_server — so importing
it has no side effects (kiosk_server runs argparse at module load; importing it
from roster_client would re-trigger that and sys.exit on foreign argv).

These helpers touch ONLY the pkl on disk. The in-memory recognition lists
(app.state.known_encodings/known_labels) stay with each caller — they are
running-server state this module knows nothing about.

The pkl holds a build_encodings.EncodingsDB with three parallel lists
(encodings/labels/meta); importing EncodingsDB here is what lets pickle
reconstruct it. build_encodings is import-safe (its argparse is guarded
under __main__).

upsert() is drop-then-append: any existing entries for the label are replaced,
never duplicated. The enroll endpoint blocks duplicate names upstream, but the
roster echo-back re-applies existing labels on every watermark crossing —
replace semantics keep that idempotent.
"""

import os
import pickle

import numpy as np

from build_encodings import EncodingsDB


def load(path: str) -> EncodingsDB:
    with open(path, "rb") as f:
        return pickle.load(f)


def save(path: str, db: EncodingsDB) -> None:
    """Write the pkl atomically: temp file + os.replace, never in place."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(db, f)
    os.replace(tmp, path)


def _drop(db: EncodingsDB, label: str) -> None:
    """Remove all entries for `label` from the three parallel lists, in place."""
    kept = [
        (enc, lbl, meta)
        for enc, lbl, meta in zip(db.encodings, db.labels, db.meta)
        if lbl != label
    ]
    if kept:
        db.encodings, db.labels, db.meta = map(list, zip(*kept))
    else:
        db.encodings, db.labels, db.meta = [], [], []


def upsert(path: str, label: str, embedding: np.ndarray, source: str) -> None:
    """Replace any existing entries for `label`, then append the fresh one."""
    db = load(path)
    _drop(db, label)
    db.encodings.append(embedding.tolist())
    db.labels.append(label)
    db.meta.append({"source": source, "label": label})
    save(path, db)


def remove(path: str, label: str) -> None:
    """Drop all entries for `label` from the pkl (no-op if absent)."""
    db = load(path)
    _drop(db, label)
    save(path, db)
