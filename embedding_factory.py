"""
Face embedding abstraction layer.

Provides a unified interface for multiple embedding backends (dlib, ArcFace)
via the FaceEmbedder Protocol and a create_embedder() factory function.

Design mirrors detector_factory.py:
  Protocol  ->  concrete classes  ->  factory function

Usage:
    embedder = create_embedder("dlib")
    vec = embedder.embed(face_crop)       # 128-D numpy array

    embedder = create_embedder("arcface")
    vec = embedder.embed(aligned_112)     # 512-D numpy array
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocol (interface contract)
# ---------------------------------------------------------------------------

class FaceEmbedder(Protocol):
    """
    Any face embedder must implement embed() and expose embedding_dim.

    embed() receives a face crop (RGB uint8 numpy array) and returns a 1-D
    float embedding vector, or None if encoding fails.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding (e.g. 128, 512)."""
        ...

    def embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute an embedding vector for a single face image.

        Args:
            face_image: RGB uint8 numpy array. Shape depends on backend:
                        - DlibEmbedder: any reasonable face crop
                        - ArcFaceEmbedder: 112x112 aligned face (from align_face())

        Returns:
            1-D float64 numpy array of length embedding_dim, or None on failure.
        """
        ...


# ---------------------------------------------------------------------------
# dlib embedder (legacy, 128-D, via face_recognition library)
# ---------------------------------------------------------------------------

class DlibEmbedder:
    """
    Wraps the dlib-based face_recognition library.

    Produces 128-D embeddings (~100-200ms per face on CPU).
    This is the original encoder used in Phases 1-2.
    """

    def __init__(self) -> None:
        import face_recognition as _fr  # lazy import
        self._fr = _fr

    @property
    def embedding_dim(self) -> int:
        return 128

    def embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode a face crop using dlib's ResNet model.

        face_recognition.face_encodings(image) runs its own internal face
        detection. Since we already have a tight crop, it usually finds
        exactly one face. If it finds none, we return None.
        """
        face_image = np.ascontiguousarray(face_image)
        try:
            encodings = self._fr.face_encodings(face_image)
        except Exception:
            return None

        if not encodings:
            return None
        return encodings[0]  # 128-D float64 numpy array


# ---------------------------------------------------------------------------
# ArcFace embedder (modern, 512-D, via InsightFace)
# ---------------------------------------------------------------------------

class ArcFaceEmbedder:
    """
    InsightFace ArcFace embedding model.

    Produces 512-D embeddings (~10-20ms per face on CPU).
    Expects 112x112 aligned face input for best accuracy (use align_face()
    from detector_factory.py).

    Works on any crop size — the model resizes internally — but alignment
    significantly improves matching quality.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1) -> None:
        """
        Args:
            model_name: InsightFace model pack name (e.g. "buffalo_l").
                        The ArcFace rec model (w600k_r50.onnx) is loaded
                        directly — no other models are loaded.
            ctx_id: -1 for CPU, 0+ for GPU device ID.
        """
        import warnings
        from pathlib import Path

        warnings.filterwarnings(
            "ignore",
            message=".*Specified provider.*not in available provider.*",
        )
        import insightface  # lazy import

        # Load only the recognition model, not the full FaceAnalysis pipeline.
        model_path = Path.home() / ".insightface" / "models" / model_name / "w600k_r50.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. "
                "Run RetinaFaceDetector first to download the buffalo_l pack, "
                "or manually download it."
            )

        self._rec_model = insightface.model_zoo.get_model(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._rec_model.prepare(ctx_id=ctx_id)

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute ArcFace embedding for a face image.

        Args:
            face_image: RGB uint8 numpy array, ideally 112x112 aligned.

        The model internally converts to BGR and normalizes. We handle the
        resize to 112x112 here if the input differs, so callers don't need
        to worry about it.
        """
        import cv2

        try:
            # Ensure 112x112 input (ArcFace's expected size)
            if face_image.shape[:2] != (112, 112):
                face_image = cv2.resize(face_image, (112, 112))

            # InsightFace rec models expect BGR input
            bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

            # get_feat expects (1, 3, 112, 112) blob — use model's get method
            # which handles preprocessing, or call get_feat directly.
            embedding = self._rec_model.get_feat(bgr)

            # get_feat returns shape (1, 512) — flatten to 1-D
            if embedding is not None:
                embedding = embedding.flatten().astype(np.float64)
                # L2-normalize so Euclidean distances are bounded [0, 2]
                # and comparable to dlib's normalized output.
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding

        except Exception:
            return None

        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(embedder_type: str, **kwargs) -> FaceEmbedder:
    """
    Instantiate a face embedder by name.

    Args:
        embedder_type: "dlib" or "arcface"
        **kwargs: Passed to the embedder constructor.
                  DlibEmbedder: (none)
                  ArcFaceEmbedder: model_name (optional), ctx_id (optional)

    Returns:
        A FaceEmbedder instance.
    """
    if embedder_type == "dlib":
        return DlibEmbedder()

    elif embedder_type == "arcface":
        model_name = kwargs.get("model_name", "buffalo_l")
        ctx_id = kwargs.get("ctx_id", -1)
        return ArcFaceEmbedder(model_name=model_name, ctx_id=ctx_id)

    else:
        raise ValueError(
            f"Unknown embedder type: '{embedder_type}'. "
            "Choose 'dlib' or 'arcface'."
        )
