"""
Interactive liveness detection for kiosk anti-spoofing.

Uses 5-point RetinaFace landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
to detect blinks and nods. A static photo cannot produce these actions.

Usage:
    manager = LivenessManager()
    session = manager.start_session("Alice", landmarks, distance)
    state, info = manager.process_frame("Alice", landmarks, frame_rgb, distance, time.time())
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ChallengeType(Enum):
    BLINK = "blink"
    NOD = "nod"


class SessionState(Enum):
    CHALLENGE_ACTIVE = "challenge_active"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class LivenessSession:
    identity: str
    challenge_type: ChallengeType
    state: SessionState
    created_at: float
    distances: List[float] = field(default_factory=list)
    # Blink tracking
    eye_metrics: List[float] = field(default_factory=list)
    baseline_eye_metric: float = 0.0
    blink_detected: bool = False
    # Nod tracking
    nod_ratios: List[float] = field(default_factory=list)
    baseline_nod_ratio: float = 0.0
    nod_detected: bool = False
    # Timeout
    timeout: float = 8.0


def extract_eye_metric(frame_rgb: np.ndarray, landmarks: np.ndarray) -> float:
    """
    Compute eye openness metric using vertical gradient energy around eye regions.

    When eyes are open, there's a strong horizontal edge (iris/eyelid boundary).
    When closed, this edge disappears → metric drops.

    Args:
        frame_rgb: Full RGB frame.
        landmarks: 5x2 array (left_eye, right_eye, nose, left_mouth, right_mouth).

    Returns:
        Average vertical gradient energy for both eyes (higher = eyes more open).
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    ied = max(np.linalg.norm(right_eye - left_eye), 1.0)

    # Eye patch size proportional to inter-eye distance
    pw = int(ied * 0.3)  # patch half-width
    ph = int(ied * 0.15)  # patch half-height

    metrics = []
    for eye_center in [left_eye, right_eye]:
        cx, cy = int(eye_center[0]), int(eye_center[1])
        y1 = max(0, cy - ph)
        y2 = min(gray.shape[0], cy + ph)
        x1 = max(0, cx - pw)
        x2 = min(gray.shape[1], cx + pw)

        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        # Sobel-Y captures horizontal edges (eyelid boundary)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        metrics.append(float(np.mean(np.abs(sobel_y))))

    return sum(metrics) / len(metrics) if metrics else 0.0


def compute_nod_ratio(landmarks: np.ndarray) -> float:
    """
    Compute nod ratio: vertical distance from eye midpoint to nose, normalized by
    inter-eye distance. Increases when head tilts down (nod).

    Args:
        landmarks: 5x2 array.

    Returns:
        Nod ratio (typically ~0.6-0.9 for neutral, changes during nod).
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    ied = max(np.linalg.norm(right_eye - left_eye), 1.0)

    return (nose[1] - eye_mid_y) / ied


class LivenessManager:
    """Manages liveness challenge sessions for the kiosk."""

    def __init__(
        self,
        challenge_timeout: float = 8.0,
        blink_threshold: float = 0.78,
        nod_threshold: float = 0.15,
    ):
        self.sessions: Dict[str, LivenessSession] = {}
        self.challenge_timeout = challenge_timeout
        self.blink_threshold = blink_threshold  # ratio of baseline that counts as "closed"
        self.nod_threshold = nod_threshold  # min nod_ratio excursion from baseline

    def start_session(
        self,
        identity: str,
        landmarks: np.ndarray,
        frame_rgb: np.ndarray,
        distance: float,
    ) -> LivenessSession:
        """Create a new liveness challenge session after identity is confirmed."""
        # Blink-only: nod detection is vulnerable to phone tilting
        challenge = ChallengeType.BLINK

        session = LivenessSession(
            identity=identity,
            challenge_type=challenge,
            state=SessionState.CHALLENGE_ACTIVE,
            created_at=time.time(),
            distances=[distance],
            timeout=self.challenge_timeout,
        )

        # Set baselines from the first frame
        session.baseline_eye_metric = extract_eye_metric(frame_rgb, landmarks)
        session.eye_metrics.append(session.baseline_eye_metric)

        session.baseline_nod_ratio = compute_nod_ratio(landmarks)
        session.nod_ratios.append(session.baseline_nod_ratio)

        # Clear any existing session and store new one
        self.sessions.clear()
        self.sessions[identity] = session

        print(f"[LIVENESS] Started {challenge.value} challenge for {identity} "
              f"(eye_baseline={session.baseline_eye_metric:.2f}, "
              f"nod_baseline={session.baseline_nod_ratio:.3f})")

        return session

    def process_frame(
        self,
        identity: str,
        landmarks: np.ndarray,
        frame_rgb: np.ndarray,
        distance: float,
    ) -> Tuple[SessionState, dict]:
        """
        Process a new frame during an active liveness challenge.

        Returns:
            (state, info_dict) where info_dict has challenge details.
        """
        session = self.sessions.get(identity)
        if session is None or session.state != SessionState.CHALLENGE_ACTIVE:
            return SessionState.FAILED, {"message": "No active session"}

        now = time.time()
        elapsed = now - session.created_at

        # Check timeout
        if elapsed > session.timeout:
            session.state = SessionState.FAILED
            print(f"[LIVENESS] {identity} challenge TIMED OUT after {elapsed:.1f}s")
            self.sessions.pop(identity, None)
            return SessionState.FAILED, {"message": "Challenge timed out"}

        session.distances.append(distance)

        # Compute current metrics
        eye_metric = extract_eye_metric(frame_rgb, landmarks)
        nod_ratio = compute_nod_ratio(landmarks)
        session.eye_metrics.append(eye_metric)
        session.nod_ratios.append(nod_ratio)

        time_remaining = max(0, session.timeout - elapsed)

        if session.challenge_type == ChallengeType.BLINK:
            result = self._check_blink(session, eye_metric)
            print(f"[LIVENESS] {identity} BLINK: eye_metric={eye_metric:.2f}, "
                  f"baseline={session.baseline_eye_metric:.2f}, "
                  f"ratio={eye_metric / max(session.baseline_eye_metric, 0.01):.2f}, "
                  f"detected={result}")
        else:
            result = self._check_nod(session, nod_ratio)
            print(f"[LIVENESS] {identity} NOD: nod_ratio={nod_ratio:.3f}, "
                  f"baseline={session.baseline_nod_ratio:.3f}, "
                  f"delta={abs(nod_ratio - session.baseline_nod_ratio):.3f}, "
                  f"detected={result}")

        if result:
            session.state = SessionState.VERIFIED
            avg_dist = sum(session.distances) / len(session.distances)
            print(f"[LIVENESS] {identity} VERIFIED via {session.challenge_type.value} "
                  f"(avg_dist={avg_dist:.4f})")
            self.sessions.pop(identity, None)
            return SessionState.VERIFIED, {
                "avg_distance": avg_dist,
                "time_remaining": time_remaining,
            }

        return SessionState.CHALLENGE_ACTIVE, {
            "challenge_type": session.challenge_type.value,
            "time_remaining": round(time_remaining, 1),
        }

    def _check_blink(self, session: LivenessSession, current_metric: float) -> bool:
        """Detect a blink: eye metric drops below threshold then recovers."""
        if session.baseline_eye_metric < 1.0:
            return False  # bad baseline

        ratio = current_metric / session.baseline_eye_metric

        # Detect the dip (eyes closing)
        if ratio < self.blink_threshold:
            session.blink_detected = True

        # Blink complete: we saw a dip and now eyes are open again
        if session.blink_detected and ratio > 0.75:
            return True

        return False

    def _check_nod(self, session: LivenessSession, current_ratio: float) -> bool:
        """Detect a nod: nose-to-eye ratio changes significantly from baseline."""
        if len(session.nod_ratios) < 3:
            return False  # need a few frames

        # Check if we've seen enough movement in nod ratio
        min_ratio = min(session.nod_ratios)
        max_ratio = max(session.nod_ratios)
        excursion = max_ratio - min_ratio

        if excursion > self.nod_threshold:
            session.nod_detected = True

        # Nod complete: we saw movement and head returned near baseline
        if session.nod_detected:
            delta_from_baseline = abs(current_ratio - session.baseline_nod_ratio)
            if delta_from_baseline < self.nod_threshold * 0.7:
                return True

        return False

    def get_session(self, identity: str) -> Optional[LivenessSession]:
        """Get active session for an identity, if any."""
        session = self.sessions.get(identity)
        if session is None:
            return None
        # Auto-expire stale sessions
        if time.time() - session.created_at > session.timeout + 2.0:
            self.sessions.pop(identity, None)
            return None
        return session

    def cleanup_stale(self, max_age: float = 15.0):
        """Remove sessions older than max_age seconds."""
        now = time.time()
        expired = [k for k, v in self.sessions.items() if now - v.created_at > max_age]
        for k in expired:
            self.sessions.pop(k, None)
