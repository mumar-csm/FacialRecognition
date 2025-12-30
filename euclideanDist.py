"""
Compares two face embeddings using Euclidean distance (and cosine similarity).
Author: Muhammed Umar

How to use in practice:
- Replace the mock embeddings with outputs from face recognition model(s).
- Adjust the threshold values based on your model's performance and requirements.
"""

import numpy as np

def l2_normalize(vec: np.ndarray, eps: float =1e-12) -> np.ndarray:
    """L2-normalizes a vector (or a batch of vectors)."""
    if vec.ndim == 1:
        norm = np.linalg.norm(vec) + eps
        return vec / norm
    elif vec.ndim == 2:
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + eps
        return vec / norms
    else:
        raise ValueError("Expected 1D or 2D array for embeddings.")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Euclidean (L2) distance between two vectors."""
    return np.linalg.norm(a - b)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a,b) / (a_norm * b_norm))

def is_same_identity_euclidean(dist: float, threshold: float) -> bool:
    """
    decodes if two embeddings belong to the same identity based on Euclidean distance.
    lower distance = higher similarity. if dist <= threshold, same identity.
    """
    return dist <= threshold

# mock embeddings for demo purposes go here
# in prod, will obtain from face recognition model(s)/face encoder
np.random.seed(42)
embedding1 = np.random.rand(512)
embedding2 = embedding1 + np.random.randn(512) * 0.05 # slight variation to simulate same identity of face A
embedding3 = np.random.randn(512)  # different identity face B

# config: normalization and threshold
USE_L2_NORMALIZATION = True

# baseline threshold suggestion"
# for -l2 normalized embeddings, Euclidean distances between same identities often fall roughly in the 0.8 to 1.2 range,
# depending on the model and dataset. different identities then to be 1.2 or much higher.
# be sure to tune on validation set.

EUCLIDEAN_THRESHOLD = 1.0 #example threshold for Euclidean distance

def compare_embeddings(a: np.ndarray, b:np.ndarray, normalize: bool = True, threshold: float = 1.0):
    # Optional normalization
    if normalize:
        a = l2_normalize(a)
        b = l2_normalize(b)
    
    #  Metrics
    dist = euclidean_distance(a, b)
    cos = cosine_similarity(a, b)

    # Decision (Euclidean)
    same = is_same_identity_euclidean(dist, threshold)
    return{
        "euclidean_distance": dist,
        "cosine_similarity": cos,
        "same_identity_by_euclidean": same,
        "threshold_used": threshold,
        "normalized": normalize
    }



def pretty_print_result(title: str, result: dict):
    print(f"\n=== {title} ===")
    print(f"Normalized: {result['normalized']}")
    print(f"Euclidean distance: {result['euclidean_distance']:.6f}")
    print(f"Cosine similarity:  {result['cosine_similarity']:.6f}  (higher -> more similar)")
    print(f"Threshold (Euclidean): {result['threshold_used']:.3f}")
    print(f"Same identity (Euclidean decision)? {'YES' if result['same_identity_by_euclidean'] else 'NO'}")

# ----------------------------
# Run comparisons
# ----------------------------
res1 = compare_embeddings(embedding1, embedding2, normalize=USE_L2_NORMALIZATION, threshold=EUCLIDEAN_THRESHOLD)
res2 = compare_embeddings(embedding1, embedding3, normalize=USE_L2_NORMALIZATION, threshold=EUCLIDEAN_THRESHOLD)

pretty_print_result("Face A vs Slightly Perturbed Face A", res1)
pretty_print_result("Face A vs Different Face", res2)