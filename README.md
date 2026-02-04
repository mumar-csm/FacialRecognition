# Facial Recognition System

An employee facial recognition application using Python, OpenCV, and dlib. The system detects faces in images or video streams, computes facial embeddings, and matches them against a pre-built database for identification.

## Features

- **Face Detection** - Uses Haar Cascade classifiers for robust face detection
- **128-D Embeddings** - Computes facial embeddings using dlib's ResNet model
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

2. **Create a virtual environment**
   ```bash
   python -m venv face_recognition_env

   # Windows
   face_recognition_env\Scripts\activate

   # Linux/macOS
   source face_recognition_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: dlib installation can be slow (compiles from source). On Windows, ensure Visual Studio Build Tools are installed first.

## Project Structure

```
FacialRecognition/
├── build_encodings.py      # Build face embeddings database
├── recognize.py            # Runtime recognition (webcam/image)
├── detect_faces.py         # Face detection utility
├── inspect_pkl.py          # Database inspection tool
├── euclideanDist.py        # Distance calculation utilities
├── requirements.txt        # Python dependencies
├── data/
│   ├── haarcascade_frontalface_default.xml   # Face detector model
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

```bash
python build_encodings.py --root ./employees --cascade data/haarcascade_frontalface_default.xml
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--root` | Directory containing employee photos | (required) |
| `--cascade` | Path to Haar Cascade XML | (required) |
| `--output` | Output database path | `data/known_faces.pkl` |
| `--margin` | Crop margin around face | `0.20` |
| `--max-long-edge` | Resize cap for large images | `1600` |
| `--rebuild` | Ignore existing DB and rebuild | `False` |
| `--verbose` | Enable debug logging | `False` |

### Step 3: Run Recognition

**Webcam Mode (Real-Time)**
```bash
python recognize.py --mode webcam --source 0
```

**Image Mode**
```bash
python recognize.py --mode image --source photo.jpg --output result.jpg
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Recognition mode: `webcam`, `image`, `video` | `webcam` |
| `--source` | Camera index or file path | `0` |
| `--database` | Path to face database | `data/known_faces.pkl` |
| `--threshold` | Match threshold (lower = stricter) | `1.0` |
| `--output` | Save annotated result (image mode) | None |
| `--resize-width` | Frame width for performance | `640` |
| `--no-display` | Disable visualization | `False` |

**Controls:**
- Press `q` to quit webcam mode

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

