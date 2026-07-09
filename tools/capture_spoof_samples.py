#!/usr/bin/env python3
"""
capture_spoof_samples.py — build the labeled real-vs-replay dataset for tuning
a screen/replay detector (Stage 1 anti-spoofing).

Captures face crops LIVE from the kiosk camera — nothing is uploaded into the
app. You generate the two buckets physically, in front of the camera:

    # people standing at the kiosk:
    python tools/capture_spoof_samples.py --label real

    # a phone/tablet PLAYING A VIDEO (or a live video call) held to the camera:
    python tools/capture_spoof_samples.py --label replay

Why live capture: the whole detection path (lens -> v4l2 exposure lock -> detector
-> inset crop) is identical to production, so the saved samples carry the exact
moire / reflection / banding artifacts a real attack produces. Feeding video files
straight into the app would skip the optics that create those artifacts.

Each capture writes:
  - data/spoof_dataset/<label>/<timestamp>.png   the inset face crop (MiniFAS input region)
  - data/spoof_dataset/labels.csv                one row per crop, with scores + metrics

The CSV is what you plot afterwards: real vs replay distributions of minifas_score
(and, later, your FFT/moire metric) tell you where a threshold actually separates them.

Controls (preview window): SPACE = capture, A = toggle auto-capture, Q = quit.
Headless: pass --auto to grab one sample every --interval seconds with no window.

Reuses the kiosk's own detector + spoof-crop helpers so samples match production.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# Same directory as kiosk_server.py so imports resolve when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector_factory import create_detector
from anti_spoof_factory import create_anti_spoof
# Single source of truth for the spoof-crop path — do NOT reimplement these here,
# or the dataset will drift from what the kiosk actually feeds the model. Lives in
# anti_spoof_crop (not kiosk_server) so importing it has no argv/model side effects.
from anti_spoof_crop import (_inset_bbox, evaluate_anti_spoof,
                             normalize_face_illumination, SPOOF_CROP_INSET)


class _SpoofState:
    """Minimal stand-in for the kiosk's app.state — evaluate_anti_spoof only
    reads .anti_spoof off it."""
    def __init__(self, anti_spoof):
        self.anti_spoof = anti_spoof


def largest_face(detections):
    """Pick the biggest bbox — the person/phone closest to the camera, which is
    the one being enrolled/attacked. Matches the kiosk's single-subject flow."""
    best, best_area = None, 0
    for bbox, _landmarks in detections:
        _x, _y, w, h = bbox
        if w * h > best_area:
            best, best_area = bbox, w * h
    return best


def parse_args():
    p = argparse.ArgumentParser(description="Capture labeled real/replay face crops for Stage 1 tuning")
    p.add_argument("--label", required=True, choices=["real", "replay"],
                   help="Which bucket to save into: real people vs. a screen playing video")
    p.add_argument("--out", default="data/spoof_dataset",
                   help="Dataset root; crops go to <out>/<label>/, CSV to <out>/labels.csv")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    # Detector defaults mirror kiosk_server.py so crops are identical to production.
    p.add_argument("--detector", choices=["haar", "retinaface", "scrfd"], default="scrfd")
    p.add_argument("--scrfd-model", dest="scrfd_model",
                   default="~/.insightface/models/buffalo_l/det_10g.onnx",
                   help="SCRFD ONNX path (same default as the kiosk)")
    p.add_argument("--det-size", type=int, default=320, dest="det_size")
    p.add_argument("--anti-spoof", dest="anti_spoof", default="minifas",
                   choices=["none", "minifas"])
    p.add_argument("--spoof-threshold", type=float, default=0.55, dest="spoof_threshold",
                   help="MiniFAS threshold (kiosk default 0.55) — only affects the logged is_real flag")
    p.add_argument("--auto", action="store_true",
                   help="Headless: auto-capture every --interval seconds, no preview window")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between auto-captures")
    p.add_argument("--count", type=int, default=0,
                   help="Stop after N captures (0 = unlimited)")
    return p.parse_args()


def build_detector(args):
    if args.detector == "scrfd":
        return create_detector("scrfd",
                               model_path=os.path.expanduser(args.scrfd_model),
                               det_size=(args.det_size, args.det_size))
    if args.detector == "retinaface":
        return create_detector("retinaface", det_size=(args.det_size, args.det_size))
    return create_detector(args.detector,
                           cascade_path="data/haarcascade_frontalface_default.xml")


def main():
    args = parse_args()

    label_dir = os.path.join(args.out, args.label)
    os.makedirs(label_dir, exist_ok=True)
    csv_path = os.path.join(args.out, "labels.csv")
    csv_is_new = not os.path.exists(csv_path)

    detector = build_detector(args)
    anti_spoof = create_anti_spoof(args.anti_spoof,
                                   threshold=args.spoof_threshold) if args.anti_spoof != "none" else None
    state = _SpoofState(anti_spoof)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera}")

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if csv_is_new:
        writer.writerow(["timestamp", "filename", "label", "minifas_is_real",
                         "minifas_score", "p_real", "lighting_ok", "x", "y", "w", "h",
                         "mean_luma", "contrast_std"])

    print(f"[capture] label={args.label}  out={label_dir}")
    print("[capture] SPACE=capture  A=toggle auto  Q=quit" if not args.auto
          else f"[capture] headless auto every {args.interval}s")

    auto = args.auto
    last_auto = 0.0
    saved = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[capture] frame grab failed", file=sys.stderr)
                break
            # Detector + anti-spoof both consume RGB (see the factory Protocols).
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            bbox = largest_face(detector.detect(frame_rgb))
            score, is_real, lighting_ok = None, None, None
            if bbox is not None and anti_spoof is not None:
                is_real, score, lighting_ok = evaluate_anti_spoof(state, frame_rgb, bbox)

            do_capture = False
            if auto and bbox is not None and (time.time() - last_auto) >= args.interval:
                do_capture, last_auto = True, time.time()

            if not args.auto:  # interactive preview
                disp = frame_bgr.copy()
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    tag = f"{args.label} n={saved}"
                    if score is not None:
                        tag += f" real={is_real} s={score:.2f} light_ok={lighting_ok}"
                    cv2.putText(disp, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                cv2.imshow("capture_spoof_samples", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
                    auto = not auto
                    last_auto = 0.0
                if key == ord(" ") and bbox is not None:
                    do_capture = True

            if do_capture:
                # Save the SAME inset crop the kiosk feeds MiniFAS (pre-normalization,
                # so a future FFT/moire detector sees the raw pixels off the sensor).
                ix, iy, iw, ih = _inset_bbox(*bbox, SPOOF_CROP_INSET, frame_rgb.shape)
                crop_rgb = frame_rgb[iy:iy + ih, ix:ix + iw]
                if crop_rgb.size == 0:
                    continue
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"{ts}.png"
                cv2.imwrite(os.path.join(label_dir, fname),
                            cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

                # Signed P(real) — the directional, tunable axis (see probability_real).
                # Feed the SAME normalized inset crop evaluate_anti_spoof classified.
                p_real = ""
                if anti_spoof is not None and hasattr(anti_spoof, "probability_real"):
                    p_real = f"{anti_spoof.probability_real(normalize_face_illumination(crop_rgb)):.4f}"

                gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
                writer.writerow([ts, fname, args.label, is_real,
                                 f"{score:.4f}" if score is not None else "",
                                 p_real, lighting_ok, ix, iy, iw, ih,
                                 f"{float(np.mean(gray)):.1f}", f"{float(np.std(gray)):.1f}"])
                csv_file.flush()
                saved += 1
                print(f"[capture] saved {args.label}/{fname}  "
                      f"score={score if score is None else round(score,3)}  total={saved}")

                if args.count and saved >= args.count:
                    print(f"[capture] reached --count {args.count}, stopping")
                    break
    finally:
        cap.release()
        if not args.auto:
            cv2.destroyAllWindows()
        csv_file.close()
        print(f"[capture] done — {saved} sample(s) in {label_dir}, log at {csv_path}")


if __name__ == "__main__":
    main()
