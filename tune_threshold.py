"""
Threshold tuning tool for the facial recognition system.

Analyzes the face encoding database to recommend optimal distance thresholds
by computing all pairwise distances between encodings and separating them
into genuine (same identity) and impostor (different identity) distributions.

Usage:
    python tune_threshold.py --database data/known_faces.pkl
"""
import argparse
import pickle
import sys
from itertools import combinations

import numpy as np

from euclideanDist import euclidean_distance
from build_encodings import EncodingsDB


def load_database(db_path):
    """Load EncodingsDB from pickle and return encodings array + labels list."""
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    encodings = np.array(db.encodings)
    labels = db.labels
    print(f"Loaded {len(labels)} encodings ({encodings.shape[1]}-D) from {db_path}")
    return encodings, labels


def compute_distances(encodings, labels):
    """
    Compute pairwise Euclidean distances between all encoding pairs.

    Returns:
        genuine_dists: distances between same-identity pairs
        impostor_dists: distances between different-identity pairs
    """
    genuine_dists = []
    impostor_dists = []

    for i, j in combinations(range(len(labels)), 2):
        dist = euclidean_distance(encodings[i], encodings[j])
        if labels[i] == labels[j]:
            genuine_dists.append(dist)
        else:
            impostor_dists.append(dist)

    return genuine_dists, impostor_dists


def recommend_threshold(genuine, impostor):
    """
    Recommend thresholds based on genuine/impostor distance distributions.

    Returns dict with threshold recommendations.
    """
    recommendations = {}

    if impostor:
        impostor_arr = np.array(impostor)
        recommendations["impostor_min"] = float(np.min(impostor_arr))
        recommendations["impostor_mean"] = float(np.mean(impostor_arr))
        recommendations["impostor_std"] = float(np.std(impostor_arr))
        # Lenient: allow most genuine through, reject only obvious impostors
        recommendations["lenient"] = float(np.percentile(impostor_arr, 0.1))

    if genuine:
        genuine_arr = np.array(genuine)
        recommendations["genuine_mean"] = float(np.mean(genuine_arr))
        recommendations["genuine_std"] = float(np.std(genuine_arr))
        # Strict: reject anything beyond worst genuine match
        recommendations["strict"] = float(np.percentile(genuine_arr, 99.9))

        if impostor:
            # EER: find threshold where FPR ~= FNR
            thresholds = np.linspace(0, max(np.max(genuine_arr), np.max(impostor_arr)), 1000)
            fnr = np.array([np.mean(genuine_arr > t) for t in thresholds])
            fpr = np.array([np.mean(impostor_arr <= t) for t in thresholds])
            eer_idx = np.argmin(np.abs(fnr - fpr))
            recommendations["eer_threshold"] = float(thresholds[eer_idx])
            recommendations["eer_value"] = float((fnr[eer_idx] + fpr[eer_idx]) / 2)

    return recommendations


def plot_distribution(genuine, impostor, output_path):
    """Plot histogram of genuine/impostor distances and ROC curve (if genuine pairs exist)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot. Install with: pip install matplotlib")
        return

    has_genuine = len(genuine) > 0
    ncols = 2 if has_genuine else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Subplot 1: Distance histogram
    ax = axes[0]
    if has_genuine:
        ax.hist(genuine, bins=50, alpha=0.7, label=f"Genuine (n={len(genuine)})", color="green")
    ax.hist(impostor, bins=50, alpha=0.7, label=f"Impostor (n={len(impostor)})", color="red")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distance Distribution")
    ax.legend()

    # Subplot 2: ROC curve (only if genuine pairs exist)
    if has_genuine:
        ax2 = axes[1]
        genuine_arr = np.array(genuine)
        impostor_arr = np.array(impostor)
        thresholds = np.linspace(0, max(np.max(genuine_arr), np.max(impostor_arr)), 500)
        tpr = np.array([np.mean(genuine_arr <= t) for t in thresholds])
        fpr = np.array([np.mean(impostor_arr <= t) for t in thresholds])
        auc = np.trapz(tpr, fpr)
        ax2.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze face database and recommend distance thresholds")
    parser.add_argument("--database", default="data/known_faces.pkl", help="Path to face encodings database")
    parser.add_argument("--output", default="threshold_analysis.png", help="Output plot path")
    args = parser.parse_args()

    # Load
    encodings, labels = load_database(args.database)
    unique_labels = set(labels)
    print(f"Unique identities: {len(unique_labels)}")

    # Compute distances
    genuine, impostor = compute_distances(encodings, labels)
    print(f"\nPairwise distances: {len(genuine)} genuine, {len(impostor)} impostor")

    if not genuine:
        print("\n[WARNING] No genuine pairs found (need multiple photos per person).")
        print("  Only impostor distances are available. Add more photos per person")
        print("  to get genuine pair analysis and EER-based threshold recommendations.")

    if not impostor:
        print("\n[ERROR] No impostor pairs found. Need at least 2 different identities.")
        sys.exit(1)

    # Statistics
    if genuine:
        g = np.array(genuine)
        print(f"\nGenuine distances:  mean={np.mean(g):.4f}  std={np.std(g):.4f}  "
              f"min={np.min(g):.4f}  max={np.max(g):.4f}")

    imp = np.array(impostor)
    print(f"Impostor distances: mean={np.mean(imp):.4f}  std={np.std(imp):.4f}  "
          f"min={np.min(imp):.4f}  max={np.max(imp):.4f}")

    # Recommendations
    recs = recommend_threshold(genuine, impostor)
    print("\n--- Threshold Recommendations ---")

    if "eer_threshold" in recs:
        print(f"  EER threshold (balanced):  {recs['eer_threshold']:.4f}  (EER={recs['eer_value']:.4f})")
    if "strict" in recs:
        print(f"  Strict (low false accept):  {recs['strict']:.4f}")
    if "lenient" in recs:
        print(f"  Lenient (low false reject): {recs['lenient']:.4f}")

    print(f"\n  Impostor minimum distance: {recs.get('impostor_min', 'N/A')}")
    if "impostor_min" in recs:
        print(f"  -> Any threshold below {recs['impostor_min']:.4f} risks false matches")

    # Plot
    plot_distribution(genuine, impostor, args.output)


if __name__ == "__main__":
    main()
