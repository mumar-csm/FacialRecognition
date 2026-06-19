# Facial Recognition System

An employee facial recognition application using Python, OpenCV, and pluggable detection/embedding backends. The system detects faces in images or video streams, computes facial embeddings, and matches them against a pre-built database for identification.

## Features

- **Face Detection** - Pluggable: Haar Cascade (fast, ~5ms) or RetinaFace (accurate, neural network)
- **Face Embedding** - Pluggable: dlib 128-D (legacy) or ArcFace 512-D (modern, better accuracy)
- **Real-Time Recognition** - Live webcam recognition with bounding boxes and confidence scores
- **Image Recognition** - Analyze static images and save annotated results
- **Flexible Input** - Supports JPG, PNG, RGB, RGBA, and grayscale images
- **Quality Control** - Validates images and enforces single-face-per-image during enrollment

## How It Works

The system operates in two phases:

### Phase 1: Enrollment (Offline)
```
Employee Images → Face Detection → Embedding Computation → Database (known_faces.pkl)
```

### Phase 2: Recognition (Runtime)
```
Camera/Image → Face Detection → Embedding → Distance Matching → Identified Employee
```

## Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- **Windows**: 
   - Visual Studio Build Tools with C++ workload (required for dlib)
   - You do NOT need Visual Studio Build Tools if you install dlib from a precompiled wheel
   - This project uses `dlib 19.24.99`, installed via a Windows precompiled wheel compatible with Python 3.12
   - I used a prebuilt wheel to avoid the Visual Studio compilation toolchain
- **Linux**: `build-essential`, `cmake`, `libopenblas-dev`
- **macOS**: Xcode Command Line Tools

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FacialRecognition.git
   cd FacialRecognition
   ```

2. **Create a conda environment** (recommended)

   dlib requires C++ compilation when installed via pip, which often fails. Using conda avoids this by providing pre-built binaries.

   ```bash
   conda create -n face python=3.12
   conda activate face
   ```

3. **Install dlib via conda**
   ```bash
   conda install -c conda-forge dlib=19.24.2
   ```

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Important**: numpy must stay below 2.0 (already pinned in `requirements.txt`). dlib 19.24.2 is incompatible with numpy 2.x due to breaking ABI changes. If you see `Unsupported image type, must be 8bit gray or RGB image` errors, run `pip install "numpy>=1.20.0,<2.0"`.

### Alternative: pip-only installation

If you prefer not to use conda, you can use a standard venv, but you must install cmake first so dlib can compile from source:

```bash
python -m venv face_recognition_env

# Windows
face_recognition_env\Scripts\activate

# Linux/macOS
source face_recognition_env/bin/activate

pip install cmake
pip install -r requirements.txt
```

> **Windows note**: You may need Visual Studio Build Tools with "Desktop development with C++" instead of cmake. Alternatively, install dlib from a precompiled wheel (e.g., `pip install dlib-19.24.99-cp312-cp312-win_amd64.whl`).

## Project Structure

```
FacialRecognition/
├── build_encodings.py      # Build face embeddings database
├── recognize.py            # Runtime recognition (webcam/image/video/RTSP)
├── kiosk_server.py         # Kiosk attendance server (pilot app, FastAPI)
├── liveness.py             # Blink-based liveness challenge
├── anti_spoof_factory.py   # Anti-spoof backends (MiniFAS)
├── detector_factory.py     # Detection backends (Haar, RetinaFace)
├── embedding_factory.py    # Embedding backends (dlib, ArcFace)
├── tracker.py              # SimpleTracker for real-time optimization
├── visualization.py        # Bounding box / label drawing
├── tune_threshold.py       # Threshold analysis tool
├── euclideanDist.py        # Distance calculation utilities
├── detect_faces.py         # Face detection utility
├── inspect_pkl.py          # Database inspection tool
├── requirements.txt        # Python dependencies
├── data/
│   ├── haarcascade_frontalface_default.xml   # Haar detector model
│   ├── known_faces.pkl                       # Face database (generated)
│   └── logs/
│       └── recognition.csv                   # Recognition event log
└── debug_scripts/          # Troubleshooting utilities
    ├── check_dlib.py
    ├── verify_db_simple.py
    └── ...
```

## Usage

### Step 1: Prepare Employee Images

Organize employee photos in a folder with this naming convention:
```
employees/
├── john_doe_1.jpg
├── john_doe_2.jpg
├── jane_smith_1.png
└── jane_smith_2.png
```

- Use underscores in names (everything before the last `_` becomes the employee ID)
- One face per image
- Clear, front-facing photos work best
- Minimum resolution: 100x100 pixels

### Step 2: Build the Face Database

**Haar + dlib (legacy)**
```bash
python build_encodings.py --root ./employees
```

**RetinaFace + ArcFace (recommended)**
```bash
python build_encodings.py --embedder arcface --detector retinaface --align --root ./employees --output data/known_faces_arcface.pkl
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--root` | Directory containing employee photos | (required) |
| `--cascade` | Path to Haar Cascade XML | `data/haarcascade_frontalface_default.xml` |
| `--output` | Output database path | `data/known_faces.pkl` |
| `--detector` | Detection backend: `haar`, `retinaface` | `haar` |
| `--embedder` | Embedding backend: `dlib`, `arcface` | `dlib` |
| `--align` | Enable face alignment (requires landmarks) | `False` |
| `--model` | InsightFace model pack name | `buffalo_l` |
| `--gpu` | GPU device ID (-1 for CPU) | `-1` |
| `--margin` | Crop margin around face | `0.20` |
| `--max-long-edge` | Resize cap for large images | `1600` |
| `--rebuild` | Ignore existing DB and rebuild | `False` |
| `--verbose` | Enable debug logging | `False` |

### Step 3: Run Recognition

**Webcam Mode (Real-Time)**
```bash
# Haar + dlib (fast, ~48 FPS on CPU)
python recognize.py --mode webcam --source 0 --threshold 0.58

# RetinaFace + ArcFace (accurate, ~4.5 FPS on CPU, GPU-ready)
python recognize.py --embedder arcface --detector retinaface --align --mode webcam --source 0 --database data/known_faces_arcface.pkl --threshold 1.0
```

**Image Mode**
```bash
python recognize.py --embedder arcface --detector retinaface --align --mode image --source photo.jpg --database data/known_faces_arcface.pkl --threshold 1.0 --output result.jpg
```

**Video Mode**
```bash
python recognize.py --embedder arcface --detector retinaface --align --mode video --source video.mp4 --database data/known_faces_arcface.pkl --threshold 1.0 --output result.mp4
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Recognition mode: `webcam`, `image`, `video` | `webcam` |
| `--source` | Camera index or file path | `0` |
| `--database` | Path to face database | `data/known_faces.pkl` |
| `--detector` | Detection backend: `haar`, `retinaface` | `haar` |
| `--embedder` | Embedding backend: `dlib`, `arcface` | `dlib` |
| `--align` | Enable face alignment | `False` |
| `--model` | InsightFace model pack name | `buffalo_l` |
| `--gpu` | GPU device ID (-1 for CPU) | `-1` |
| `--threshold` | Match threshold (lower = stricter) | `1.0` |
| `--output` | Save annotated result | None |
| `--resize-width` | Frame width for performance | `640` |
| `--no-display` | Disable visualization | `False` |

> **Note**: The `--embedder` must match the database. Using an ArcFace database with the dlib embedder (or vice versa) will produce an error — rebuild the database with the matching embedder.

**Controls:**
- Press `q` to quit webcam mode

### RTSP Stream Testing (Local)

To test RTSP mode locally without an IP camera, use [MediaMTX](https://github.com/bluenviron/mediamtx) and [FFmpeg](https://ffmpeg.org/) to simulate a live stream from a video file.

**Prerequisites:**
```bash
brew install mediamtx ffmpeg
```

**Setup:**

1. Create a `mediamtx.yml` config file in the project directory:
   ```yaml
   paths:
     all_others:
       source: publisher
   ```

2. **Terminal 1** — Start MediaMTX: (run it from the directory where the mediamtx.yml file lives)
   ```bash
   mediamtx
   ```
   You should see listeners open on RTSP (:8554), RTMP (:1935), and other ports.

3. **Terminal 2** — Stream a video file on loop via RTMP: (run it from the directory where the video file in question lives)
   ```bash
   ffmpeg -re -stream_loop -1 -i test_video.mp4 -c:v libx264 -f flv rtmp://localhost:1935/live
   ```
   - `-re` reads the file at real-time speed
   - `-stream_loop -1` loops the video indefinitely
   - `-c:v libx264` re-encodes to H.264 (required for FLV/RTMP)

4. **Terminal 3** — Run recognition against the RTSP output:
   ```bash
   python recognize.py --mode webcam --source "rtsp://localhost:8554/live" --threshold 0.7
   ```

MediaMTX bridges RTMP input to RTSP output on the same path name (`live`). The recognition app auto-detects the RTSP URL and uses the RTSP handler with reconnection logic.

> **Note**: FPS will match your source video's frame rate (e.g., 20 FPS source = 20 FPS recognition). When the ffmpeg stream is stopped, the app will attempt reconnection up to 5 times before exiting gracefully.

## Kiosk Attendance Server (Pilot App)

`kiosk_server.py` is the tablet/Pi attendance kiosk: it runs the full pipeline (detect → anti-spoof → align → embed → match → identity consensus → blink liveness challenge → clock-in/out) behind a FastAPI web UI. This is the app used in the in-store pilot.

> **Tip:** activate the environment first (`conda activate face`), or prefix any command with `conda run --no-capture-output -n <env> ...`.

**Start the server (defaults)**
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl
```

Then open the web UI in a browser:

| URL | Page |
|-----|------|
| `http://127.0.0.1:8000/` | Kiosk (clock-in/out) |
| `http://127.0.0.1:8000/enroll` | Enroll a new employee from the camera |
| `http://127.0.0.1:8000/manage` | View / delete employees |
| `http://127.0.0.1:8000/report` | Attendance report + CSV export |

> By default the server binds to loopback (`127.0.0.1`), so it's only reachable from the same machine. To open it to other devices on the LAN, add `--host 0.0.0.0` (only do this on a trusted network).

### Common Run Configurations

**Testing — short cooldown (clock in/out repeatedly without waiting)**

The default 120 s cooldown blocks duplicate clock events, which is painful when testing. Drop it to a few seconds so you can repeatedly clock the same person:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --cooldown 5
```

**Fast test run — easier identity confirmation**

Lower the consensus frame count and give a longer window for the blink challenge while you're getting set up:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl \
  --cooldown 5 --consensus 1 --challenge-timeout 15
```

**Disable the MiniFAS pre-screen**

Anti-spoofing has two independent layers: the MiniFAS per-frame pre-screen and the blink liveness challenge. `--anti-spoof none` turns off **only** MiniFAS — the **blink challenge still runs** and remains the primary gate, so photos are still blocked. (K2 testing found blink is the layer that actually stops phone-photo attacks; MiniFAS scores high-quality photos as "real".) Useful for removing MiniFAS borderline false-rejects while testing, without losing photo protection:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --anti-spoof none
```

**Stricter / looser face matching**

`--threshold` is the distance cutoff (lower = stricter, fewer false matches; higher = more lenient). For an ArcFace database, ~0.6 is the working default:
```bash
# Stricter — reject borderline matches
python kiosk_server.py --database data/known_faces_arcface.pkl --threshold 0.5

# Looser — accept more distant matches (more false accepts)
python kiosk_server.py --database data/known_faces_arcface.pkl --threshold 0.75
```

**Stricter anti-spoof**

`--spoof-threshold` is the "real face" probability cutoff (0–1, higher = stricter). Raise it if photo attacks are slipping through:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --spoof-threshold 0.7
```

**Faster pipeline on weak hardware (e.g. Raspberry Pi) — recommended Pi config**

Use `--detector scrfd` and shrink the detector input size. `scrfd` is the **same** detection network as the default `retinaface`, but loaded directly — it skips the `FaceAnalysis` auxiliary models (106-point landmarks, 3D landmarks, gender/age) that run every frame and the kiosk never uses, so it's faster with no loss of accuracy or landmark quality. `--det-size 320` is the default; 256 or 224 trades a little accuracy for more speed:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --detector scrfd
```

> `scrfd` expects the buffalo_l model already on disk (`~/.insightface/models/buffalo_l/det_10g.onnx`) and errors if it's missing. It's present once you've built the ArcFace database (Step 2 downloads it). The default `retinaface` auto-downloads buffalo_l on first run, which is why it stays the safe out-of-box default — but on a Pi or in production, `scrfd` is the better choice.

**Manager-PIN-gated enrollment**

Require a PIN before anyone can enroll or delete employees via the web UI:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --enrollment-pin 4821
```

**Local timezone for reports**

Report timestamps default to UTC. Set the store's timezone so the report page reads correctly:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl --timezone America/New_York
```

**Dev sync/roster intervals (central HQ wiring)**

Production drains the outbox and pulls the roster every 30 min. For local testing against a central server, point at it and tighten the intervals so changes propagate in seconds:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl \
  --central-url http://localhost:9000 --store-id downtown-01 \
  --sync-interval-seconds 30 --roster-interval-seconds 30
# API key is read from the CENTRAL_API_KEY env var, never a flag
```

**POS punch bridge (type the employee's POS ID into Oracle on clock-in/out)**

When a Teensy POS bridge is attached (see [pos_bridge/](pos_bridge/)), point the kiosk at its serial device and every successful clock-in/out types the recognized employee's 7-digit POS ID into the focused POS terminal:
```bash
python kiosk_server.py --database data/known_faces_arcface.pkl \
  --pos-serial-port /dev/cu.usbmodem12345
```

> Punching is **best-effort**: attendance is always recorded even if the Teensy is unplugged or the port is wrong — those just log a warning. Omit `--pos-serial-port` to disable it entirely. Employees without a POS ID are skipped.

### Kiosk Server Options

| Flag | Description | Default |
|------|-------------|---------|
| `--database` | Face encodings pkl file | `data/known_faces_arcface.pkl` |
| `--sqlite` | Attendance/spoof SQLite DB | `data/kiosk.db` |
| `--threshold` | Face-match distance cutoff (lower = stricter) | `0.6` |
| `--cooldown` | Seconds between duplicate clock events | `120` |
| `--consensus` | Consecutive frames to confirm identity before liveness | `3` |
| `--spoof-threshold` | Anti-spoof "real" probability cutoff (higher = stricter) | `0.55` |
| `--challenge-timeout` | Seconds allowed for the blink challenge | `8.0` |
| `--anti-spoof` | Anti-spoof backend: `minifas`, `none` | `minifas` |
| `--detector` | Detection backend: `retinaface`, `scrfd`, `haar`. `scrfd` is the same model as `retinaface` with less overhead — recommended on a Pi (see above) | `retinaface` |
| `--det-size` | Detector input size (smaller = faster) | `320` |
| `--embedder` | Embedding backend: `arcface`, `dlib` | `arcface` |
| `--enrollment-pin` | Manager PIN to gate enroll/delete | None |
| `--timezone` | Local timezone for report timestamps | `UTC` |
| `--camera-id` | Identifier for this kiosk instance | `kiosk-01` |
| `--retention-days` | Attendance record retention | `365` |
| `--spoof-retention-days` | Spoof-attempt record retention | `90` |
| `--store-id` / `--device-id` | Location / Pi identifiers (multi-store sync) | host fallback |
| `--device-config` | JSON file with device/store/central settings | None |
| `--central-url` | Central HQ base URL (omit to disable upload) | None |
| `--sync-interval-seconds` | Outbox drain interval (prod 1800) | 1800 |
| `--roster-interval-seconds` | Roster pull interval (prod 1800) | 1800 |
| `--pos-serial-port` | Teensy POS punch bridge serial device (omit to disable; see [pos_bridge/](pos_bridge/)) | None |
| `--pos-baud` | Baud for the POS serial bridge (must match the Teensy sketch) | `115200` |
| `--host` / `--port` | Bind address / port | `127.0.0.1` / `8000` |

> The `CENTRAL_API_KEY` for HQ uploads is read from the environment only, never passed as a flag.

## Utilities

### Test Face Detection
```bash
# Test on an image
python detect_faces.py --image photo.jpg

# Test with webcam
python detect_faces.py --webcam
```

### Inspect Database
```bash
python inspect_pkl.py data/known_faces.pkl
```

## Troubleshooting

### Debug Scripts

| Script | Purpose |
|--------|---------|
| `debug_scripts/check_dlib.py` | Verify dlib installation and model loading |
| `debug_scripts/verify_db_simple.py` | Validate database integrity and encoding quality |
| `debug_scripts/test_encoding.py` | Test the encoding pipeline on a single image |
| `debug_scripts/debug_image.py` | Debug face detection on a specific image |

### Common Issues

**dlib installation fails**
- Windows: (recommended method from windows - no compilation) Install Visual Studio Build Tools with "Desktop development with C++"
   - this project uses `dlib 19.24.99` installed form a precompiled Windows wheel that matches Python 3.12 and a 64 bit architecture (win_amd64)
   - to install the same version, run `pip install dlib-19.24.99-cp312-cp312-win_amd64.whl`
- Linux: `sudo apt install build-essential cmake libopenblas-dev`, `pip install dlib`
- Try: `pip install cmake` first, then `pip install dlib`

**No faces detected**
- Ensure good lighting and front-facing images
- Try adjusting `--min-neighbors` (lower = more detections, more false positives)
- Use `detect_faces.py` to test detection on your images

**Poor recognition accuracy**
- Add more enrollment photos per person (different angles, lighting)
- Lower the `--threshold` value (e.g., `0.6`) for stricter matching
- Ensure enrollment photos are clear and well-lit

**"Unknown" for known faces**
- Rebuild the database: `python build_encodings.py --rebuild ...`
- Check database with: `python debug_scripts/verify_db_simple.py`

## Configuration

Key parameters in `build_encodings.py` that can be tuned:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `scale_factor` | Haar Cascade scale factor | `1.1` |
| `min_neighbors` | Haar Cascade min neighbors | `5` |
| `min_size` | Minimum face size in pixels | `(60, 60)` |
| `crop_margin` | Margin around detected face | `0.20` |

### Threshold Tuning

Use `tune_threshold.py` to analyze your face database and find the optimal `--threshold` value:

```bash
pip install matplotlib scikit-learn
python tune_threshold.py --database data/known_faces.pkl
```

This compares all pairwise distances between encodings and outputs:
- **Impostor distances** (different people) — statistics and distribution histogram
- **Genuine distances** (same person, different photos) — requires multiple photos per person
- **Recommended thresholds** — strict, balanced (EER), and lenient options

**Current findings** (11 employees, 1 photo each — impostor-only analysis):

| Embedder | Impostor Range | Recommended Threshold |
|----------|---------------|----------------------|
| dlib (128-D) | 0.56 – 1.01 | ≤ **0.56** |
| ArcFace (512-D) | 1.18 – 1.52 | ≤ **1.18** |

ArcFace provides 2x better impostor separation with tighter variance (std 0.06 vs 0.09).
Re-run after adding multiple photos per person for full EER-based recommendations.

The tool saves a `threshold_analysis.png` plot for visual inspection.

