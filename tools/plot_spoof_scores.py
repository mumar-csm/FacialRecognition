#!/usr/bin/env python3
"""
plot_spoof_scores.py — eyeball real-vs-replay separation from the capture harness.

Reads data/spoof_dataset/labels.csv (written by capture_spoof_samples.py) and,
for each numeric signal, prints how well `real` separates from `replay` and the
threshold that best splits them. Run it as you capture to see whether a signal
is actually usable before building a detector around it.

    python tools/plot_spoof_scores.py
    python tools/plot_spoof_scores.py --plot   # also save histograms PNG (needs matplotlib)

Also reports mean_luma / contrast_std separation — if THOSE separate as well as
minifas_score, your buckets differ by brightness/framing (the trap), not screen-ness.
"""

from __future__ import annotations

import argparse
import csv
import os

# Signals worth checking. p_real (signed, directional) is the real candidate;
# minifas_score is the old direction-stripped confidence (kept for comparison);
# mean_luma / contrast_std are the nuisance check — they SHOULD overlap between
# buckets if you captured well.
METRICS = ["p_real", "minifas_score", "mean_luma", "contrast_std"]


def load(csv_path):
    rows = {"real": {m: [] for m in METRICS}, "replay": {m: [] for m in METRICS}}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            label = r.get("label")
            if label not in rows:
                continue
            for m in METRICS:
                try:
                    rows[label][m].append(float(r[m]))
                except (KeyError, ValueError):
                    pass
    return rows


def stats(vals):
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    return {"n": n, "min": vals[0], "max": vals[-1], "mean": mean,
            "median": vals[n // 2]}


def best_threshold(real, replay):
    """Threshold that maximizes correct split, plus its accuracy. Assumes real
    tends higher than replay (flips automatically if not)."""
    if not real or not replay:
        return None
    cand = sorted(set(real + replay))
    best_t, best_acc, best_dir = None, 0.0, ">="
    for t in cand:
        for direction in (">=", "<"):
            if direction == ">=":
                correct = sum(v >= t for v in real) + sum(v < t for v in replay)
            else:
                correct = sum(v < t for v in real) + sum(v >= t for v in replay)
            acc = correct / (len(real) + len(replay))
            if acc > best_acc:
                best_t, best_acc, best_dir = t, acc, direction
    return {"threshold": best_t, "accuracy": best_acc, "rule": f"real {best_dir} {best_t:.3f}"}


def main():
    ap = argparse.ArgumentParser(description="Report real-vs-replay separation from labels.csv")
    ap.add_argument("--csv", default="data/spoof_dataset/labels.csv")
    ap.add_argument("--plot", action="store_true", help="Also save histograms PNG (needs matplotlib)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"No CSV at {args.csv} — capture some samples first.")

    data = load(args.csv)
    nr, np_ = len(data["real"]["minifas_score"]), len(data["replay"]["minifas_score"])
    print(f"samples: real={nr}  replay={np_}\n")

    for m in METRICS:
        rs, ps = stats(data["real"][m]), stats(data["replay"][m])
        if not rs or not ps:
            print(f"{m}: not enough data\n")
            continue
        print(f"── {m} ──")
        print(f"  real   n={rs['n']:<4} mean={rs['mean']:.3f}  median={rs['median']:.3f}  range=[{rs['min']:.3f},{rs['max']:.3f}]")
        print(f"  replay n={ps['n']:<4} mean={ps['mean']:.3f}  median={ps['median']:.3f}  range=[{ps['min']:.3f},{ps['max']:.3f}]")
        bt = best_threshold(data["real"][m], data["replay"][m])
        if bt:
            flag = "  <- separates (nuisance? check framing)" if m != "minifas_score" and bt["accuracy"] >= 0.8 else ""
            print(f"  best split: {bt['rule']}  acc={bt['accuracy']:.0%}{flag}")
        print()

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("--plot needs matplotlib (pip install matplotlib)")
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
        for ax, m in zip(axes, METRICS):
            ax.hist(data["real"][m], bins=20, alpha=0.6, label="real")
            ax.hist(data["replay"][m], bins=20, alpha=0.6, label="replay")
            ax.set_title(m); ax.legend()
        out = os.path.join(os.path.dirname(args.csv), "score_histograms.png")
        fig.tight_layout(); fig.savefig(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
