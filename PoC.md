# Facial Recognition System - PoC

**Project Goal**: Build a kiosk-based facial recognition clock-in/out system for fast food restaurants (~30 stores, ~30 employees per store) to prevent ghost employees.

**Approach**: Incremental delivery — Phases 1-4 built the core recognition engine (video, RTSP, tracker, multi-camera, RetinaFace, ArcFace). Phases K1-K3 pivot to a kiosk-based attendance system with anti-spoofing, a tablet-friendly web UI, and audit reporting.

**Architecture**: Tablet (browser) → captures frame via webcam → sends to local server → recognition pipeline → green/red result. Option 3 (audit-based logging) with upgrade path to kiosk lock enforcement.

---

## 📋 Overall Progress Tracker

### Core Engine (Complete)
- [x] Phase 1: Video & RTSP Support
- [x] Phase 2: CPU Performance Optimizations
- [x] Phase 3: Library Modernization (RetinaFace + ArcFace)
- [x] Phase 4: Multi-Camera NVR Support

### Kiosk Pivot
- [ ] Phase K1: Anti-Spoofing Integration
- [ ] Phase K2: Kiosk Recognition App
- [ ] Phase K3: Enrollment UI & Audit Reports

### Deferred
- [ ] Phase 5: Optional FAISS Indexing (Future)
- [ ] Phase 6: GPU Acceleration (When GPU Available)

---

## Phase 1: Quick Wins - Video & RTSP Support (Week 1)

**Goal**: Enable video file processing and RTSP/IP camera streams with minimal code changes.

### 1.1 Video File Support (2 days)

#### Tasks:
- [x] **Add `recognize_from_video()` function** in [recognize.py](recognize.py)
  - [x] Load face database using `load_database()`
  - [x] Open video file with `cv2.VideoCapture(video_path)`
  - [x] Get video properties (FPS, total frames, resolution)
  - [x] Implement frame reading loop with frame skip logic
  - [x] Add progress reporting: "Processing frame X/Y (Z%)"
  - [x] Process each frame using existing `process_frame()` function
  - [x] Create video writer for annotated output using `cv2.VideoWriter()`
  - [x] Write fourcc codec matching input video
  - [x] Collect statistics: total frames, faces detected, unique identities
  - [x] Return statistics dictionary

- [x] **Add `--frame-skip` CLI argument** in [recognize.py](recognize.py) parse_args()
  - [x] Add argument: `--frame-skip` type=int, default=0
  - [x] Document: "Process every Nth frame (0=all, 1=every other, 2=every 3rd)"

- [x] **Update video mode handler** in [recognize.py](recognize.py) main()
  - [x] Replace error message with call to `recognize_from_video()`
  - [x] Pass all required arguments: source, database, threshold, cascade, output, frame_skip, resize_width
  - [x] Print summary statistics after processing
  - [x] Format: total frames, faces detected, unique identities, processing time

#### Testing Checklist:
- [x] Test with short video (5-10 seconds)
- [x] Test with .mp4 file (H.264 codec)
- [ ] Test with .avi file (MJPEG codec)
- [x] Test frame skip: `--frame-skip 0` (all frames)
- [x] Test frame skip: `--frame-skip 1` (every other frame)
- [x] Test frame skip: `--frame-skip 2` (every 3rd frame)
- [ ] Verify output video is playable in VLC
- [x] Verify annotations are correct (bounding boxes, labels)
- [x] Measure FPS difference with different skip values

#### Success Criteria:
- [x] Video mode works without errors
- [x] Output video has correct annotations
- [x] Frame skipping reduces processing time proportionally
- [x] Statistics are accurate and informative

---

### 1.2 RTSP Stream Support (3 days)

#### Tasks:
- [x] **Add `recognize_from_rtsp()` function** in [recognize.py](recognize.py) (after video function)
  - [x] Load face database using `load_database()`
  - [x] Set FFMPEG environment variable: `OPENCV_FFMPEG_CAPTURE_OPTIONS`
  - [x] Configure: "rtsp_transport;tcp|timeout;10000000"
  - [x] Open RTSP stream with `cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)`
  - [x] Set buffer size: `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` for low latency
  - [x] Check if stream opened successfully
  - [x] Implement reconnection logic with retry attempts
  - [x] Add frame reading loop with error handling
  - [x] Process frames using existing `process_frame()` function
  - [x] Display with FPS counter
  - [x] Handle stream disconnection gracefully
  - [x] Exit on 'q' key press

- [x] **Implement reconnection logic**
  - [x] Track reconnection attempts counter
  - [x] Sleep 2 seconds between reconnection attempts
  - [x] Release and recreate VideoCapture on failure
  - [x] Reset attempt counter on successful reconnection
  - [x] Exit after max attempts reached

- [x] **Add `--tracker-interval` CLI argument** in [recognize.py](recognize.py) parse_args()
  - [x] Add argument: `--tracker-interval` type=int, default=30
  - [x] Document: "Frames between re-identification (default: 30, ~1 second at 30 FPS)"
  - [x] Pass to SimpleTracker(reidentify_interval=args.tracker_interval) in all modes

- [x] **Extend webcam mode to auto-detect RTSP** in [recognize.py](recognize.py) main() (lines 850-904)
  - [x] Check if source starts with "rtsp://"
  - [x] If RTSP URL detected, call `recognize_from_rtsp()` instead of webcam function
  - [x] Otherwise, parse as integer camera index and proceed normally

#### Testing Checklist:
- [x] Test with local RTSP stream (using MediaMTX + FFmpeg)
- [x] Verify stream displays correctly
- [x] Verify face recognition works on RTSP stream
- [ ] Test with authenticated RTSP (username:password in URL)
- [x] Test reconnection: unplug network cable mid-stream
- [x] Verify automatic reconnection works
- [ ] Test with real IP camera (if available)
- [x] Measure latency from stream to display

#### Success Criteria:
- [x] RTSP streams connect successfully
- [ ] Authentication works (if applicable)
- [x] Reconnection logic recovers from network interruptions
- [x] Latency is acceptable (<500ms with buffer=1)
- [x] Face recognition accuracy matches webcam mode

#### RTSP URL Format Reference:
```
rtsp://[username:password@]host[:port]/path

Examples:
- rtsp://192.168.1.100:554/stream1
- rtsp://admin:password@10.0.0.50:8554/cam/realmonitor?channel=1
- rtsp://nvr.local:554/Streaming/Channels/101
```

---

## Phase 2: CPU Performance Optimizations (Week 2)

**Goal**: Maximize real-time performance on CPU with multiprocessing and smarter frame processing.

### 2.1 Frame Skipping & Smart Processing (2 days)

#### Tasks:
- [x] **Create `SimpleTracker` class** in [recognize.py](recognize.py) (after Detection class)
  - [x] Add `__init__` with reidentify_interval parameter (default: 30 frames)
  - [x] Add `last_boxes` list to store previous frame bounding boxes
  - [x] Add `last_labels` list to store previous frame identities
  - [x] Add `last_confidences` list to store previous confidences
  - [x] Add `frames_since_identify` counter
  - [x] Implement `compute_iou(box1, box2)` method
    - [x] Calculate intersection area
    - [x] Calculate union area
    - [x] Return IoU ratio
  - [x] Implement `should_reidentify(current_boxes)` method
    - [x] Increment frames_since_identify counter
    - [x] Return True if counter >= reidentify_interval
    - [x] Check IoU between current and previous boxes
    - [x] Return True if boxes moved significantly (IoU < 0.5)
    - [x] Otherwise return False
  - [x] Implement `update(boxes, labels, confidences)` method
    - [x] Store current frame data as "last" data
    - [x] Reset counter if re-identification occurred

- [x] **Integrate tracker in webcam mode** [recognize.py](recognize.py)
  - [x] Create SimpleTracker instance before loop
  - [x] In loop: detect faces every frame (Haar is fast)
  - [x] Check if re-identification needed via `tracker.should_reidentify()`
  - [x] If True: encode and match faces as normal
  - [x] If False: reuse cached labels and confidences
  - [x] Update tracker with current frame data

- [x] **Add tracker to RTSP mode**
  - [x] Duplicate tracker integration in RTSP function

#### Testing Checklist:
- [x] Test webcam with tracker enabled
- [x] Verify identities remain stable when person is stationary
- [x] Verify re-identification occurs when person moves
- [x] Verify re-identification occurs every 30 frames (1 second at 30 FPS)
- [x] Measure FPS before tracker: baseline
- [x] Measure FPS after tracker: should be 3-5x higher
- [x] Test with multiple faces in frame
- [x] Verify no crashes or memory leaks during long runs

#### Success Criteria:
- [x] Real-time FPS improves by 3-5x
- [x] Identity labels remain stable for stationary faces
- [x] System re-identifies when faces move significantly
- [x] No degradation in recognition accuracy

---

### 2.2 Multiprocessing for Enrollment (1 day)

#### Tasks:
- [x] **Create worker function** in [build_encodings.py](build_encodings.py) (before cli_main)
  - [x] Define `encode_single_image(args)` function
  - [x] Unpack args: (image_record, config_dict)
  - [x] Move existing encoding logic from cli_main() into this function
  - [x] Load and validate image
  - [x] Preprocess image
  - [x] Detect faces
  - [x] Validate single face
  - [x] Encode face
  - [x] Return FaceRecord or skip-reason string on error
  - [x] Ensure all errors are caught and logged

- [x] **Modify cli_main()** in [build_encodings.py](build_encodings.py)
  - [x] Import: `from multiprocessing import Pool, cpu_count`
  - [x] Prepare work items: list of (image_record, config_dict) tuples
  - [x] Calculate num_workers: `max(1, cpu_count() - 1)` (leave one core free)
  - [x] Print: "Encoding N images using M workers..."
  - [x] Create Pool: `with Pool(num_workers) as pool:`
  - [x] Execute: `pool.imap_unordered(encode_single_image, work_items)`
  - [x] Collect results and count skip reasons
  - [x] Continue with existing serialization logic

- [x] **Handle progress reporting**
  - [x] Used `pool.imap_unordered()` with in-place progress counter
  - [x] Removed worker debug prints to keep progress output clean

#### Testing Checklist:
- [x] Verify: output .pkl has same number of records and labels as sequential version
- [x] Test with small dataset (10 images): verify correctness (10 encoded, 0 skipped)
- [x] No crashes or deadlocks
- [ ] Time sequential vs parallel (deferred — not meaningful with 10 images)
- [ ] Test with large dataset (100+ images) when available
- [ ] Monitor CPU usage with larger dataset

#### Success Criteria:
- [ ] Enrollment time reduced by 4-8x (needs larger dataset to measure)
- [x] Output database matches sequential version (same record count and labels)
- [x] No crashes or deadlocks
- [ ] CPU utilization is high during processing (needs larger dataset to observe)

---

### 2.3 Threshold Tuning Tool (2 days)

> **Scope note**: This is a one-time analysis tool, not a runtime dependency. Consider implementing as a Jupyter notebook instead of a CLI script to avoid adding matplotlib/scikit-learn as permanent project dependencies. The output (recommended threshold value) is what matters — the tool itself only needs to run once per database rebuild.

#### Tasks:
- [x] **Create tune_threshold.py script** (CLI script)
  - [x] Add docstring
  - [x] Import: pickle, numpy, matplotlib, euclideanDist
  - [x] Add `compute_distances(encodings, labels)` function
    - [x] All-pairs comparison using itertools.combinations
    - [x] Separate into genuine_dists and impostor_dists
    - [x] Return both lists
  - [x] Add `plot_distribution(genuine, impostor, output_path)` function
    - [x] Histogram of genuine and impostor distances (bins=50, alpha=0.7)
    - [x] ROC curve with AUC (when genuine pairs available)
    - [x] Graceful fallback if matplotlib not installed
  - [x] Add `recommend_threshold(genuine, impostor)` function
    - [x] EER threshold (when genuine pairs available)
    - [x] Strict threshold: 99.9th percentile of genuine
    - [x] Lenient threshold: 0.1st percentile of impostor
    - [x] Handles edge case: no genuine pairs (impostor-only analysis)
  - [x] Add main section with argparse, stats, recommendations, plot

- [x] **Update requirements.txt**
  - [x] Add: `matplotlib>=3.5.0`
  - [x] Add: `scikit-learn>=1.0.0`

#### Testing Checklist:
- [x] Run: `python tune_threshold.py --database data/known_faces.pkl`
- [x] Verify: no errors
- [x] Check output: threshold_analysis.png created
- [x] Open plot: verify histogram shows impostor distribution
- [ ] Check ROC curve: AUC >0.95 (deferred — needs genuine pairs)
- [ ] Verify EER recommendation (deferred — needs genuine pairs)
- [x] Document recommended threshold in README

#### Success Criteria:
- [x] Script runs without errors
- [x] Plots are clear and informative
- [x] Recommendations are data-driven and actionable
- [x] User can choose threshold based on use case (strict vs lenient)

---

## Phase 3: Library Modernization - CPU Optimized (Weeks 3-4)

**Goal**: Replace Haar Cascades and dlib with modern libraries that are faster on CPU and GPU-ready for future.

### 3.1 Detection Factory with RetinaFace/MTCNN (1 week)

#### Tasks:
- [x] **Install new dependencies**
  - [x] Update requirements.txt: `insightface>=0.7.0`
  - [x] Update requirements.txt: `onnxruntime>=1.15.0`
  - [x] Update requirements.txt: `scikit-image>=0.19.0`
  - [x] Run: `pip install insightface onnxruntime scikit-image`
  - [x] Verify installation: `python -c "import insightface; print(insightface.__version__)"`

- [x] **Create detector_factory.py** (new file)
  - [x] Add imports: typing, Protocol, numpy, cv2, insightface
  - [x] Define `FaceDetector` Protocol class
    - [x] Method: `detect(image) -> List[Tuple[bbox, landmarks]]`
    - [x] bbox format: (x, y, w, h)
    - [x] landmarks format: 5x2 numpy array or None
  - [x] Implement `HaarDetector` class
    - [x] Init: load cascade, set parameters
    - [x] detect(): convert to grayscale, detectMultiScale, return results
    - [x] Return format: [(bbox, None), ...] (no landmarks)
  - [x] Implement `RetinaFaceDetector` class
    - [x] Init: load InsightFace model, set providers (CPU/GPU)
    - [x] Use: `FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])`
    - [x] Call: `app.prepare(ctx_id=-1)` for CPU
    - [x] detect(): call `app.get(image)`, parse results
    - [x] Convert bbox from [x1,y1,x2,y2] to [x,y,w,h]
    - [x] Return format: [(bbox, landmarks), ...]
  - [x] Implement `create_detector(detector_type, **kwargs)` factory
    - [x] Support: "haar", "retinaface"
    - [x] Return appropriate detector instance
  - [x] Add `align_face(image, landmarks, output_size)` utility
    - [x] Use scikit-image SimilarityTransform
    - [x] Define standard reference points for 112x112 output
    - [x] Compute transform from landmarks to reference
    - [x] Warp image using cv2.warpAffine
    - [x] Return aligned face image

- [x] **Update recognize.py to use detector factory**
  - [x] Import: `from detector_factory import create_detector, align_face`
  - [x] Add `--detector` argument in parse_args()
    - [x] choices=["haar", "retinaface"], default="haar"
  - [x] Add `--align` flag in parse_args()
    - [x] action="store_true", help="Enable face alignment"
  - [x] Modify `detect_and_encode_faces()` function
    - [x] Accept detector object instead of cascade_path
    - [x] Call detector.detect(frame) instead of Haar cascade
    - [x] Iterate through detections (bbox, landmarks)
    - [x] If align flag and landmarks available: align face
    - [x] Otherwise: extract ROI as before
    - [x] Encode face and collect results
  - [x] Update `recognize_from_webcam()`
    - [x] Create detector: `detector = create_detector(args.detector, ...)`
    - [x] Pass detector to detect_and_encode_faces()
  - [x] Update `recognize_from_image()`
    - [x] Create detector
    - [x] Pass detector to detect_and_encode_faces()
  - [x] Update `recognize_from_video()`
    - [x] Create detector
    - [x] Pass detector to detect_and_encode_faces()
  - [x] Update `recognize_from_rtsp()`
    - [x] Create detector
    - [x] Pass detector to detect_and_encode_faces()

- [x] **Update build_encodings.py to use detector factory**
  - [x] Import: `from detector_factory import create_detector, align_face`
  - [x] Add `--detector` argument in parse_args()
  - [x] Add `--align` flag in parse_args()
  - [x] Modify detect_faces() call in encode_single_image() worker
    - [x] Create detector instance (lazy per-worker cache)
    - [x] Call detector.detect() instead of cv2 cascade
  - [x] Update encoding logic to use alignment if enabled

#### Testing Checklist:
- [x] Test Haar detector (backward compatibility)
  - [x] Run: `python recognize.py --detector haar --mode webcam`
  - [x] Verify: works as before (~30 FPS)
- [x] Test RetinaFace detector
  - [x] Run: `python recognize.py --detector retinaface --mode image`
  - [x] Verify: detects faces (7 faces detected vs fewer with Haar)
  - [x] Compare detection quality visually
- [x] Test face alignment
  - [x] Run: `python recognize.py --detector retinaface --align --mode image --source test.jpg --threshold 0.58`
  - [x] Compared: aligned (0.410) vs non-aligned (0.401) — no meaningful difference with dlib encoder (expected, alignment benefits ArcFace in Phase 3.2)
- [x] Rebuild database with RetinaFace
  - [x] Run: `python build_encodings.py --detector retinaface --align --root "../Office Team Profile Pics"`
  - [x] Built both aligned and non-aligned DBs for comparison
- [x] Performance comparison
  - [x] Measure FPS: Haar (~30 FPS) vs RetinaFace (~5 FPS) on CPU
  - [x] RetinaFace ~6x slower — expected for neural network inference on CPU
- [x] Side-by-side comparison
  - [x] Same image, both detectors: Haar distance=0.40 (wider bbox), RetinaFace distance=0.42 (tighter bbox)
  - [x] RetinaFace bbox more precisely fitted around face; slight distance increase expected with dlib encoder

#### Success Criteria:
- [x] Haar detector still works (backward compatibility)
- [x] RetinaFace provides better detection accuracy
- [x] Face alignment verified — no improvement with dlib encoder (expected); plumbing ready for ArcFace in Phase 3.2
- [x] No crashes or errors with either detector
- [x] Code is clean and maintainable with factory pattern

---

### 3.2 Embedding Factory with ArcFace (1 week)

#### Tasks:
- [x] **Create embedding_factory.py** (new file)
  - [x] Add imports: typing, Protocol, Optional, numpy, insightface
  - [x] Define `FaceEmbedder` Protocol class
    - [x] Method: `embed(face_image) -> Optional[np.ndarray]`
    - [x] Property: `embedding_dim() -> int`
  - [x] Implement `DlibEmbedder` class
    - [x] Init: import face_recognition library
    - [x] embed(): call face_recognition.face_encodings()
    - [x] embedding_dim: return 128
  - [x] Implement `ArcFaceEmbedder` class
    - [x] Init: load InsightFace ArcFace model directly via `model_zoo.get_model()` (not FaceAnalysis)
    - [x] Loads only `w600k_r50.onnx` rec model — no wasted detection/landmark/genderage models
    - [x] Call: `model.prepare(ctx_id)` (-1 for CPU, 0+ for GPU)
    - [x] embed(): resize face to 112x112, call model.get_feat()
    - [x] embedding_dim: return 512
  - [x] Implement `create_embedder(embedder_type, **kwargs)` factory
    - [x] Support: "dlib", "arcface"
    - [x] Return appropriate embedder instance

- [x] **Update EncodingsDB schema** in [build_encodings.py](build_encodings.py)
  - [x] Add field: `embedding_dim: int = 128`
  - [x] Add field: `embedder_type: str = "dlib"`
  - [x] Keep backward compatibility with v1 databases

- [x] **Update database loader** in [recognize.py](recognize.py)
  - [x] Modify `load_database()` function
  - [x] Check for embedder_type field (use hasattr with default)
  - [x] Check for embedding_dim field (use hasattr with default)
  - [x] Print info: "Database: {embedder_type} ({embedding_dim}-D embeddings)"
  - [x] Warn if CLI embedder != database embedder

- [x] **Update recognize.py to use embedding factory**
  - [x] Import: `from embedding_factory import create_embedder`
  - [x] Add `--embedder` argument in parse_args()
    - [x] choices=["dlib", "arcface"], default="dlib"
  - [x] Add `--model` argument in parse_args()
    - [x] default="buffalo_l"
    - [x] help="InsightFace model pack name"
  - [x] Add `--gpu` argument in parse_args()
    - [x] type=int, default=-1
    - [x] help="GPU device ID (-1 for CPU)"
  - [x] In main(): load database and check embedder compatibility
    - [x] Warn if args.embedder != db_embedder
    - [x] Create embedder: `embedder = create_embedder(args.embedder, ...)`
  - [x] Modify detect_and_encode_faces() to accept embedder
    - [x] Call embedder.embed(face_roi) instead of face_recognition
    - [x] Handle None return value

- [x] **Update build_encodings.py to use embedding factory**
  - [x] Import: `from embedding_factory import create_embedder`
  - [x] Add `--embedder` argument
  - [x] Add `--model`, `--gpu` arguments
  - [x] Create embedder instance in cli_main() (per-worker cache via _get_worker_embedder)
  - [x] Pass embedder to encoding functions
  - [x] Update serialize() to save embedder_type and embedding_dim

#### Testing Checklist:
- [x] Test dlib embedder (backward compatibility)
  - [x] Run: `python build_encodings.py --embedder dlib --root "../Office Team Profile Pics" --output data/known_faces_dlib.pkl`
  - [x] Verify: database created with 128-D embeddings (11 encoded, 1 skipped)
- [x] Test ArcFace embedder (CPU)
  - [x] Run: `python build_encodings.py --embedder arcface --detector retinaface --align --root "../Office Team Profile Pics" --output data/known_faces_arcface.pkl`
  - [x] Verify: database created with 512-D embeddings (11 encoded, 1 skipped)
  - [x] Note: uses w600k_r50.onnx from buffalo_l pack (only rec model loaded)
- [x] Test ArcFace with different models
  - [x] Skipped: PoC checklist referenced arcface_mnet_v1/arcface_r50_v1 (standalone model names from older InsightFace API). Implementation uses buffalo_l model pack with w600k_r50.onnx. Only one model pack available.
- [x] Compare embedding quality
  - [x] Rebuild database with dlib: saved as known_faces_dlib.pkl
  - [x] Rebuild database with ArcFace: saved as known_faces_arcface.pkl
  - [x] Run tune_threshold.py on both
  - [x] dlib: impostor mean=0.82, min=0.56, std=0.09, threshold ≤0.56
  - [x] ArcFace: impostor mean=1.40, min=1.18, std=0.06, threshold ≤1.18
  - [x] ArcFace has 2x better separation (min impostor 1.18 vs 0.56) and tighter std
  - [ ] EER comparison deferred (needs genuine pairs — multiple photos per person)
- [x] Test recognition with ArcFace
  - [x] Run: `python recognize.py --embedder arcface --detector retinaface --align --mode image --source "../Office Team Profile Pics/Ali_L.png" --database data/known_faces_arcface.pkl --threshold 1.0 --output result_arcface.jpg`
  - [x] Verify: Ali_L matched at distance=0.000, confidence=1.000
  - [x] Video benchmark (289 frames, test_video.mp4):
    - ArcFace+RetinaFace: 4.5 FPS, 578 faces detected, 64.9s
    - dlib+Haar: 47.8 FPS, 199 faces detected, 6.0s
    - RetinaFace detection is the bottleneck (~10x slower than Haar on CPU), not embedding
    - ArcFace embedding itself is faster but masked by detector cost
    - GPU acceleration (Phase 6) would eliminate this bottleneck
- [x] Test mixed embedder error
  - [x] Use dlib database with arcface embedder
  - [x] Verify: hard error with actionable message (not just a warning)
  - [x] Changed from warning to ValueError — 512-D vs 128-D can't compute distance
  - [x] Bug fixed: original warning let execution continue, crashed with numpy broadcast error

#### Bugs Fixed During Testing:
- `--cascade` was required in build_encodings.py even with `--detector retinaface` — changed to optional with default
- ArcFace embeddings weren't L2-normalized — raw Euclidean distances were ~25-35, now bounded 0-2
- Mixed embedder warning → hard error (incompatible dimensions crash numpy)

#### Success Criteria:
- [x] dlib embedder still works (backward compatibility)
- [x] ArcFace embedding is faster per face (~10-20ms vs ~100-200ms), but RetinaFace detection dominates total pipeline time on CPU. Net: 4.5 FPS (RetinaFace+ArcFace) vs 47.8 FPS (Haar+dlib). GPU needed to unlock embedding speedup.
- [x] ArcFace provides better matching accuracy (2x impostor separation, 3x more faces detected)
- [x] GPU path is ready (just change ctx_id when GPU available)
- [x] Database schema tracks embedder type for compatibility

---

## Phase 4: Multi-Camera NVR Support (Week 5)

**Goal**: Process 2-4 RTSP streams simultaneously for comprehensive coverage with centralized logging.

### 4.1 Parallel Multi-Stream Processing

#### Tasks:
- [x] **Create recognize_multi.py** (new file) — Process-per-camera architecture (not Pool)
  - [x] Per-process detector/embedder/tracker initialization
  - [x] RTSP stream processing with reconnection logic (5 retries, 2s delay)
  - [x] SQLite logging with WAL mode (no lock conflicts)
  - [x] Dedup logging (same identity suppressed within `--log-interval`)
  - [x] Process manager: signal handling, monitor loop, auto-restart (max 3 per camera)
  - [x] Full CLI matching recognize.py flags + `--config`, `--sqlite`, `--retention-days`, `--log-interval`
  - [x] Retention purge on startup (`purge_old_detections`)
  - [x] `init_database` called before purge to ensure table exists

- [x] **Create cameras_example.json** (new file)
  - [x] JSON structure with "cameras" array
  - [x] Each camera: name, rtsp_url, location

- [x] **Create SQLite schema**
  - [x] Table: detections (id, timestamp, camera_name, location, identity, confidence, distance, bbox_x/y/w/h)
  - [x] Indexes on timestamp, identity, camera_name
  - [x] WAL mode enabled for safe concurrent writes

- [x] **Query examples** (tested in Test 6)
  - [x] Recent detections (ORDER BY timestamp DESC)
  - [x] Unique visitors today (GROUP BY identity)
  - [x] Activity by camera (GROUP BY camera_name)

#### Testing Checklist (2026-03-24, MediaMTX + FFmpeg local RTSP):
- [x] Test 1: Single camera — detections logged, identified EMAD_L, LINA_L, TENILLE_D at threshold 0.70
- [x] Test 2: Multi-camera (2 cams) — both cameras logged, no SQLite lock errors (Front Door 27, Back Door 29 detections)
- [x] Test 3: Reconnection — worker reconnected after FFmpeg restart, resumed logging
- [x] Test 4: Ctrl+C graceful shutdown — clean exit, no tracebacks
- [x] Test 5: Retention purge — Ghost row (2025-01-01) purged, today's rows preserved
- [x] Test 6: Query examples — all 3 queries returned sensible results
- [ ] Test with 4 cameras (deferred — needs more RTSP streams or real cameras)
- [ ] Long-term stability test (deferred — 30+ min run with memory monitoring)

#### Bugs Found & Fixed:
- `purge_old_detections` called before `init_database` → "no such table: detections". Fixed by calling `init_database` in `run_multi()` before purge.

#### Success Criteria:
- [x] 2 cameras run simultaneously without issues
- [x] Reconnection logic works reliably
- [x] SQLite logging is accurate and complete
- [x] Queries are fast and informative
- [ ] 4-camera test and long-term stability (deferred)

---

## Phase K1: Anti-Spoofing Integration

**Goal**: Integrate liveness detection so employees can't clock in with a photo of someone else. This is the riskiest unknown — validate it works before building the kiosk UI.

**Background**: InsightFace buffalo_l does NOT include anti-spoofing. Need a separate model. Top candidates from research: MiniFASNet (~98% accuracy) and Silent-Face-Anti-Spoofing (~99% accuracy). Both are lightweight CNNs that classify a face crop as real/spoof.

### K1.1 Anti-Spoof Model Integration

#### Tasks:
- [ ] **Evaluate and select anti-spoofing library**
  - [ ] Test Silent-Face-Anti-Spoofing (https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
    - [ ] Clone repo, run inference on a test image
    - [ ] Measure latency per frame on M4 Mac
    - [ ] Test with phone screen photo (print attack)
    - [ ] Test with real face (should pass)
  - [ ] If Silent-Face doesn't work well, try MiniFASNet as fallback
  - [ ] Choose based on: accuracy, latency (<50ms target), ease of integration

- [ ] **Create anti_spoof_factory.py** (new file, follows existing factory pattern)
  - [ ] Define `AntiSpoofChecker` Protocol class
    - [ ] Method: `check(face_image, bbox) -> Tuple[bool, float]` (is_real, confidence)
  - [ ] Implement selected model as a class (e.g., `SilentFaceChecker`)
    - [ ] Init: load ONNX model, set providers (CPU/GPU)
    - [ ] check(): preprocess crop, run inference, return (is_real, score)
  - [ ] Implement `NoopChecker` class (always returns True — for backward compat / testing)
  - [ ] Implement `create_anti_spoof(method, **kwargs)` factory function

- [ ] **Integrate into recognition pipeline**
  - [ ] Add anti-spoof check in `detect_and_encode_faces()` (recognize.py)
    - [ ] After face detection, before embedding
    - [ ] If spoof detected: skip embedding, mark as "SPOOF" in Detection result
    - [ ] Log spoof attempts (for audit trail)
  - [ ] Add `--anti-spoof` CLI flag in recognize.py parse_args()
    - [ ] choices=["none", "silent-face"], default="none"
  - [ ] Keep backward compatible: `--anti-spoof none` skips the check entirely

#### Testing Checklist:
- [ ] Real face (webcam): passes liveness, recognized correctly
- [ ] Phone screen showing photo: detected as spoof, rejected
- [ ] Printed photo: detected as spoof, rejected
- [ ] Laptop screen showing photo: detected as spoof, rejected
- [ ] Multiple real faces in frame: all pass liveness individually
- [ ] Performance: anti-spoof adds <50ms per face on CPU
- [ ] `--anti-spoof none` still works (no regression)

#### Success Criteria:
- [ ] Photo attacks (phone/print/screen) rejected >95% of the time
- [ ] Real faces accepted >98% of the time
- [ ] Latency acceptable for kiosk use (<50ms per face)
- [ ] Clean factory pattern, consistent with detector/embedder factories

---

## Phase K2: Kiosk Recognition App

**Goal**: Build a tablet-friendly web app for clock-in/out. Employee faces camera → system recognizes + liveness check → green/red result → attendance logged.

**Architecture**: FastAPI backend (serves API + static frontend). Runs on a local machine on the store network. Tablet connects via browser.

### K2.1 Recognition API

#### Tasks:
- [ ] **Create kiosk_server.py** (new file — FastAPI app)
  - [ ] On startup: load face database, create detector, embedder, anti-spoof checker
    - [ ] Use RetinaFace + ArcFace + anti-spoof as defaults
    - [ ] Configurable via environment variables or config file
  - [ ] **POST /api/recognize** endpoint
    - [ ] Accept: base64-encoded JPEG frame (from browser webcam)
    - [ ] Pipeline: decode → detect face → anti-spoof check → embed → match → log
    - [ ] Enforce single-face: reject if 0 or >1 faces detected
    - [ ] Return JSON: `{status, identity, confidence, distance, is_live, message}`
    - [ ] Status values: "recognized", "unknown", "spoof_detected", "no_face", "multiple_faces"
  - [ ] **GET /api/health** endpoint
    - [ ] Return: model status, database info (employee count, embedder type)
  - [ ] **GET /api/attendance** endpoint
    - [ ] Return: today's attendance log (list of clock-in events)
    - [ ] Query param: `?date=YYYY-MM-DD` for other dates

- [ ] **Create SQLite attendance schema**
  - [ ] Table: `attendance` (id, timestamp, identity, confidence, distance, is_clock_in, camera_id)
  - [ ] Index on timestamp, identity
  - [ ] Dedup: suppress duplicate logs for same person within configurable window (e.g., 5 min)
  - [ ] Reuse WAL mode pattern from recognize_multi.py

- [ ] **Add cooldown logic**
  - [ ] After successful recognition, suppress same person for N minutes (configurable)
  - [ ] Prevents accidental double clock-in
  - [ ] In-memory dict: `{identity: last_seen_timestamp}`

#### Testing Checklist:
- [ ] POST valid frame with known face → "recognized" + correct identity
- [ ] POST frame with unknown face → "unknown"
- [ ] POST frame with no face → "no_face"
- [ ] POST frame with multiple faces → "multiple_faces"
- [ ] POST spoof (phone photo) → "spoof_detected"
- [ ] Duplicate clock-in within cooldown → suppressed
- [ ] GET /api/health → returns model info
- [ ] GET /api/attendance → returns today's log
- [ ] Server handles concurrent requests without crashing

#### Success Criteria:
- [ ] End-to-end recognition in <500ms per request (detect + anti-spoof + embed + match)
- [ ] Correct JSON responses for all scenarios
- [ ] Attendance logged to SQLite accurately
- [ ] Cooldown prevents double-logging

---

### K2.2 Kiosk Web Frontend

#### Tasks:
- [ ] **Create static/ directory** with kiosk UI files
  - [ ] `index.html` — single-page kiosk interface
  - [ ] `kiosk.js` — webcam capture + API interaction
  - [ ] `kiosk.css` — tablet-friendly styling

- [ ] **Webcam integration** (kiosk.js)
  - [ ] Use `navigator.mediaDevices.getUserMedia()` for camera access
  - [ ] Show live camera feed in `<video>` element
  - [ ] Auto-capture: take snapshot every N seconds (e.g., 2s) while active
    - [ ] Or manual: "Clock In" button triggers capture
    - [ ] Decide based on UX testing — start with button, switch to auto if better
  - [ ] Capture frame to canvas, convert to base64 JPEG
  - [ ] POST to `/api/recognize`
  - [ ] Display result

- [ ] **Result display**
  - [ ] Recognized: green overlay/animation + "Welcome, [Name]!" + timestamp
  - [ ] Unknown: red overlay + "Not recognized — see manager"
  - [ ] Spoof detected: red overlay + "Please use your real face"
  - [ ] No face: neutral prompt "Position your face in the frame"
  - [ ] Auto-reset to camera feed after 3-4 seconds

- [ ] **Tablet-friendly design**
  - [ ] Full-screen layout, large text, high contrast
  - [ ] Works in landscape and portrait
  - [ ] No scrolling needed — everything visible at once
  - [ ] Touch-friendly buttons (minimum 48px tap targets)

- [ ] **Serve frontend from FastAPI**
  - [ ] Mount static/ directory via `app.mount("/", StaticFiles(...))`
  - [ ] Root URL serves kiosk UI

#### Testing Checklist:
- [ ] Opens in mobile browser (Chrome/Safari) on tablet
- [ ] Camera permission prompt appears and works
- [ ] Live camera feed is visible
- [ ] Capture + recognize flow works end-to-end
- [ ] Green/red result displays correctly
- [ ] Auto-resets after result display
- [ ] Works in both landscape and portrait orientation
- [ ] Usable without keyboard (touch only)

#### Success Criteria:
- [ ] Non-technical employee can clock in without instructions
- [ ] Full flow (approach → capture → result) takes <5 seconds
- [ ] Clear visual feedback for all outcomes
- [ ] Works on iPad/Android tablet in browser

---

## Phase K3: Enrollment UI & Audit Reports

**Goal**: Store managers can enroll new employees via web UI and view attendance reports. Replaces CLI-only enrollment with something a non-technical user can operate.

### K3.1 Web-Based Enrollment

#### Tasks:
- [ ] **POST /api/enroll** endpoint (in kiosk_server.py)
  - [ ] Accept: base64 JPEG photo + employee name/ID
  - [ ] Pipeline: decode → detect face → enforce single face → anti-spoof → embed → save
  - [ ] Save photo to `data/employees/{employee_id}.jpg`
  - [ ] Rebuild face database (append new encoding to existing .pkl)
    - [ ] Hot-reload: update in-memory database without server restart
  - [ ] Return: success/failure + reason

- [ ] **DELETE /api/enroll/{employee_id}** endpoint
  - [ ] Remove employee from database
  - [ ] Delete photo file
  - [ ] Hot-reload database

- [ ] **GET /api/employees** endpoint
  - [ ] Return list of enrolled employees (name, enrollment date, photo thumbnail)

- [ ] **Enrollment web page** (static/enroll.html)
  - [ ] Simple form: employee name/ID + capture photo from webcam
  - [ ] Or: upload existing photo
  - [ ] Show confirmation with detected face crop
  - [ ] Manager-only access (basic password protection for PoC — proper auth later)

#### Testing Checklist:
- [ ] Enroll new employee via web UI → appears in employee list
- [ ] New employee can immediately clock in (hot-reload works)
- [ ] Reject enrollment photo with 0 or >1 faces
- [ ] Reject spoof photo during enrollment
- [ ] Delete employee → can no longer clock in
- [ ] Password protection prevents unauthorized enrollment

#### Success Criteria:
- [ ] Store manager can enroll a new employee in <1 minute
- [ ] No CLI or technical knowledge required
- [ ] Database stays consistent after add/delete operations

---

### K3.2 Attendance Reports

#### Tasks:
- [ ] **GET /api/report** endpoint
  - [ ] Query params: `?start=YYYY-MM-DD&end=YYYY-MM-DD&employee=NAME`
  - [ ] Return: attendance records (JSON)
  - [ ] Include: timestamp, identity, confidence, clock-in/out status

- [ ] **GET /api/report/csv** endpoint
  - [ ] Same filters as above
  - [ ] Return: CSV download for manual POS comparison
  - [ ] Columns: Date, Time, Employee, Confidence, Event Type

- [ ] **Report web page** (static/report.html)
  - [ ] Date range picker
  - [ ] Filter by employee (dropdown)
  - [ ] Table view of attendance records
  - [ ] "Export CSV" button
  - [ ] Summary stats: total clock-ins today, unique employees, flagged events

- [ ] **Anomaly flags** (basic)
  - [ ] Flag: employee clocked in at unusual time (outside configured shift window)
  - [ ] Flag: spoof attempts (logged from K1)
  - [ ] Flag: unrecognized face attempts (potential unauthorized person)
  - [ ] These are informational flags, not enforcement — matches Option 3 approach

#### Testing Checklist:
- [ ] Report page shows attendance data for selected date range
- [ ] Employee filter works correctly
- [ ] CSV export downloads valid file
- [ ] Anomaly flags appear for spoof attempts
- [ ] Report loads quickly even with 1000+ records

#### Success Criteria:
- [ ] Manager can view "who clocked in today" in <10 seconds
- [ ] CSV export is compatible with Excel for manual POS comparison
- [ ] Anomaly flags surface suspicious activity without false alarm noise

---

## Phase 5: Optional FAISS Indexing (Future Enhancement)

**Status**: Deferred - Not needed for 50-200 employees. Current brute-force search is fast enough (~5-10ms).

**When to implement**: If database grows beyond 100 employees and search becomes a bottleneck.

### Tasks (Future):
- [ ] Install faiss-cpu: `pip install faiss-cpu>=1.7.0`
- [ ] Create database_index.py module
- [ ] Implement FaceIndex class with FAISS indexing
- [ ] Add --use-faiss flag to build_encodings.py
- [ ] Add --use-faiss flag to recognize.py
- [ ] Test: compare brute-force vs FAISS search times
- [ ] Benchmark: measure speedup for large databases (1000+ faces)

**Expected Gain**: 5-10x search speedup for databases >500 employees. Marginal benefit for smaller databases.

---

## Phase 6: GPU Acceleration (When GPU Available)

**Status**: GPU-ready architecture in Phase 3. Just need to change flags when GPU available.

**Prerequisites**:
- [ ] NVIDIA GPU (GTX 1060 or better)
- [ ] CUDA Toolkit 11.x installed
- [ ] cuDNN 8.x installed

### Tasks:
- [ ] **Install GPU packages**
  - [ ] Uninstall: `pip uninstall onnxruntime`
  - [ ] Install: `pip install onnxruntime-gpu>=1.15.0`
  - [ ] Optional: `pip install faiss-gpu>=1.7.0`
  - [ ] Verify: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
  - [ ] Expect: ['CUDAExecutionProvider', 'CPUExecutionProvider']

- [ ] **Test GPU detection**
  - [ ] Run: `python recognize.py --embedder arcface --gpu 0 --mode webcam`
  - [ ] Verify: no CUDA errors
  - [ ] Check: GPU utilization in nvidia-smi

- [ ] **Benchmark CPU vs GPU**
  - [ ] Create benchmark_gpu.py script
  - [ ] Test 100 face encodings on CPU (ctx_id=-1)
  - [ ] Test 100 face encodings on GPU (ctx_id=0)
  - [ ] Measure time for both
  - [ ] Calculate speedup ratio
  - [ ] Print results

- [ ] **Rebuild database with GPU**
  - [ ] Run: `python build_encodings.py --embedder arcface --gpu 0`
  - [ ] Measure time: compare to CPU version
  - [ ] Expect: 5-10x speedup for enrollment

- [ ] **Test real-time recognition with GPU**
  - [ ] Run: `python recognize.py --embedder arcface --gpu 0 --mode webcam`
  - [ ] Measure FPS: should be 60+ FPS
  - [ ] Compare to CPU: should be 10-20x improvement

#### Testing Checklist:
- [ ] GPU detection works (CUDA provider available)
- [ ] Encoding speed: CPU (40-50ms) → GPU (1-2ms) = 50-100x speedup
- [ ] Real-time FPS: CPU (30-50 FPS) → GPU (60+ FPS)
- [ ] Enrollment time: CPU (30-60s) → GPU (10-20s) for 100 images
- [ ] No CUDA errors or crashes
- [ ] Falls back to CPU if GPU unavailable

#### Success Criteria:
- [ ] 50-100x speedup for face encoding
- [ ] Real-time processing at 60+ FPS
- [ ] Enrollment completes in under 30 seconds for 200 employees
- [ ] System is GPU-accelerated but CPU-compatible

---

## 🎯 Performance Targets Summary

### Original Baseline (Before Any Optimizations):
- ⏱️ Face encoding: 100-200ms per face (dlib)
- 📹 Real-time FPS: ~5 FPS (single camera, no tracker)
- 📊 Enrollment: ~5 minutes for 100 images
- 🔍 Detection: Haar Cascade (medium accuracy)

### After Phase 1.1 + 2.1 (Video + SimpleTracker) — ACTUAL:
- ✅ Video processing: Working with SimpleTracker optimization
- ⚡ Real-time FPS: **~88-89 FPS on M4 Mac** (1920x1080→640x360, threshold 0.6-0.7)
- ⚡ Real-time FPS: **~19 FPS baseline on Windows laptop** (before SimpleTracker)
- ⚡ SimpleTracker reduces encoding calls ~30x (only re-identifies every 30 frames)

### After Phase 1.2 (RTSP) — Target:
- ✅ RTSP streams: Working with reconnection
- ✅ SimpleTracker integrated in RTSP mode

### After Phase 2.2-2.3 (Multiprocessing + Threshold Tuning) — Target:
- ⚡ Enrollment: ~30-60 seconds for 100 images (4-8x improvement via multiprocessing)
- 🎯 Data-driven threshold recommendation

### After Phase 3 (Library Modernization) — Target:
- 🚀 Face encoding: 10-20ms per face (ArcFace vs 100-200ms dlib)
- 🚀 Encoding speedup compounds with SimpleTracker (faster re-identify frames)
- 🎯 Detection: RetinaFace (high accuracy + landmarks for alignment)
- 📈 Matching accuracy: 99.80% vs 99.38%

### After Phase 6 (GPU Acceleration) — Target:
- 🔥 Face encoding: 1-2ms per face (50-100x vs original dlib)
- 🔥 Real-time FPS: 60+ FPS per camera
- 🔥 Enrollment: ~10-20 seconds for 100 images

---

## 📦 Dependency Summary

### Phase 1 (Video/RTSP):
No new dependencies - OpenCV already supports RTSP

### Phase 2 (Optimizations):
```
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

### Phase 3 (Library Modernization):
```
insightface>=0.7.0
onnxruntime>=1.15.0
scikit-image>=0.19.0
```

### Phase 6 (GPU - when available):
```
onnxruntime-gpu>=1.15.0  # Replaces onnxruntime
faiss-gpu>=1.7.0  # Optional
```

---

## 🔍 Key Files Reference

### Core Recognition:
- [`recognize.py`](recognize.py) - Runtime recognition (webcam/image/video/RTSP)
- [`build_encodings.py`](build_encodings.py) - Enrollment pipeline
- [`euclideanDist.py`](euclideanDist.py) - Distance metrics

### New Modules (Phase 3+):
- `detector_factory.py` - Detection abstraction (Haar/RetinaFace)
- `embedding_factory.py` - Embedding abstraction (dlib/ArcFace)
- `tune_threshold.py` - Threshold optimization tool
- `recognize_multi.py` - Multi-camera NVR support
- `cameras.json` - Camera configuration

### Documentation:
- [`README.md`](README.md) - Project overview
- `PoC.md` - This file (implementation checklist)
- `.claude/plans/stateful-nibbling-swing.md` - Detailed plan

---

## ✅ Validation Checklist

### After Each Phase:
- [ ] All tests pass
- [ ] Performance targets met
- [ ] Backward compatibility maintained
- [ ] Documentation updated
- [ ] Code committed to git with clear message

### Before Moving to Next Phase:
- [ ] Review implementation with user
- [ ] Address any issues or bugs
- [ ] Validate on real data (not just test cases)
- [ ] Update this checklist with actual results

---

## 🐛 Troubleshooting Guide

### Common Issues:

#### RTSP Connection Fails:
- [ ] Check URL format: `rtsp://[user:pass@]host:port/path`
- [ ] Verify network connectivity: `ping <camera-ip>`
- [ ] Test stream with VLC first
- [ ] Check firewall rules
- [ ] Try TCP transport flag

#### Low FPS Performance:
- [ ] Reduce resize_width (try 480 or 320)
- [ ] Increase frame skip (try --frame-skip 2)
- [ ] Check CPU usage in task manager
- [ ] Ensure Phase 2 optimizations are active
- [ ] Consider Phase 3 library upgrades

#### Database Loading Errors:
- [ ] Verify database file exists
- [ ] Check file permissions
- [ ] Try rebuilding database
- [ ] Verify Python version compatibility (3.8+)

#### Out of Memory Errors:
- [ ] Reduce number of simultaneous cameras
- [ ] Increase frame skip rate
- [ ] Reduce resize resolution
- [ ] Check for memory leaks (restart process periodically)

---

## 📝 Notes & Observations

### Implementation Notes:
- Document any deviations from plan
- Record actual performance numbers
- Note any bugs or issues encountered
- Track time spent per phase

### Known Issues & Trade-offs:

#### Double face detection on re-identify frames (minor inefficiency)
When `should_reidentify()` returns True, `tracker.detect_faces()` has already run detection, but then `process_frame()` runs `detect_and_encode_faces()` which detects again from scratch. This means face detection happens twice on re-identify frames. A proper fix would be an encode-only function that accepts pre-detected boxes, skipping the redundant detection — but `process_frame()` is also used by image mode where there's no tracker, so it can't simply lose its detection step. Since re-identification only triggers every ~30 frames (once per second at 30 FPS) and detection is ~5-10ms, this wastes ~5-10ms per second. Not nothing, but not critical either. Worth addressing if we refactor the detection/encoding pipeline in Phase 3.

### Performance Results:
- Phase 1 + 2.1 (Video + SimpleTracker): ~88-89 FPS on M4 Mac (test_video.mp4, 1920x1080→640x360, threshold 0.6-0.7). ~19 FPS baseline on Windows laptop before SimpleTracker.
- Phase 2: (Fill in after completion)
- Phase 3: (Fill in after completion)
- Phase 4.1 (Multi-Camera NVR): 2 cameras processing simultaneously via local RTSP (MediaMTX + FFmpeg). SQLite WAL mode, no lock conflicts. 6/6 core tests passed. Process-per-camera architecture with auto-restart.

### Lessons Learned:
- What worked well?
- What was more difficult than expected?
- What would you do differently?

---

## 🔄 Implementation Progress - Option 2 (SimpleTracker + RTSP Combined)

### Completed (Commit 1 - 2026-02-19):
- ✅ **SimpleTracker class** (~170 lines)
  - Intelligent IoU-based face tracking
  - Re-identification triggers: interval (30 frames), movement (IoU < 0.5), face count change
  - Reduces encoding overhead 3-5x (5 FPS → 15-25 FPS expected)

### Completed (2026-02-26 - Bug fixes & cleanup):
- ✅ **SimpleTracker integrated into webcam mode** — tracker used in `recognize_from_webcam()`
- ✅ **`recognize_from_video()` function** — full video processing with SimpleTracker optimization
- ✅ **`--frame-skip` CLI argument** — configurable frame skipping
- ✅ **Video mode handler** — wired up in `main()` with all arguments
- ✅ **Bug fix: division by zero** — guarded `elapsed_time` in video summary
- ✅ **Bug fix: video writer empty output** — writer now uses computed output dimensions matching resize
- ✅ **Bug fix: CAP_DSHOW on macOS** — removed Windows-only backend, let OpenCV auto-select
- ✅ **Removed dead code** — `detect_faces_only()` (superseded by `SimpleTracker.detect_faces()`)
- ✅ **Extracted `DEFAULT_DETECTOR_PARAMS`** — single module-level constant replacing 3 inline dicts

### Completed (2026-03-03):
- ✅ **recognize_from_rtsp()** with SimpleTracker integration
- ✅ **Reconnection logic** — 5 retries with 2s delay, graceful exit
- ✅ **Auto-detect RTSP URLs** in main() webcam mode
- ✅ **--tracker-interval CLI argument** — configurable re-id interval
- ✅ **--max-retries CLI argument** — configurable reconnection attempts
- ✅ **Pin numpy <2.0** — dlib 19.24.2 incompatible with numpy 2.x ABI
- ✅ **README updated** — conda setup, numpy note, RTSP local testing guide

### Upcoming:
- [ ] Phase 1.2 testing checklist (public RTSP, authenticated, reconnection tests)

---

**Last Updated**: 2026-03-25
**Status**: Phases 1-4 complete (core engine). Pivoting to kiosk-based clock-in/out system.
**Next Action**: Phase K1 (Anti-Spoofing Integration)
