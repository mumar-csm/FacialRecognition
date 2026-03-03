# Facial Recognition System - NVR Integration & Optimization PoC

**Project Goal**: Add NVR/RTSP camera support, video processing, and optimize performance for 2-4 cameras with 50-200 employees.

**Approach**: Incremental delivery - Quick wins first (get NVR working), then gradual modernization (better libraries), designed for CPU with GPU-ready architecture.

**Timeline**: 5 weeks phased implementation

---

## 📋 Overall Progress Tracker

- [ ] Phase 1: Video & RTSP Support (Week 1)
- [ ] Phase 2: CPU Performance Optimizations (Week 2)
- [ ] Phase 3: Library Modernization (Weeks 3-4)
- [ ] Phase 4: Multi-Camera NVR Support (Week 5)
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
- [ ] Test with public RTSP stream: `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4`
- [ ] Verify stream displays correctly
- [ ] Verify face recognition works on RTSP stream
- [ ] Test with authenticated RTSP (username:password in URL)
- [ ] Test reconnection: unplug network cable mid-stream
- [ ] Verify automatic reconnection works
- [ ] Test with real IP camera (if available)
- [ ] Measure latency from stream to display

#### Success Criteria:
- [ ] RTSP streams connect successfully
- [ ] Authentication works (if applicable)
- [ ] Reconnection logic recovers from network interruptions
- [ ] Latency is acceptable (<500ms with buffer=1)
- [ ] Face recognition accuracy matches webcam mode

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
- [ ] Test with multiple faces in frame
- [x] Verify no crashes or memory leaks during long runs

#### Success Criteria:
- [ ] Real-time FPS improves by 3-5x
- [ ] Identity labels remain stable for stationary faces
- [ ] System re-identifies when faces move significantly
- [ ] No degradation in recognition accuracy

---

### 2.2 Multiprocessing for Enrollment (1 day)

#### Tasks:
- [ ] **Create worker function** in [build_encodings.py](build_encodings.py) (before cli_main)
  - [ ] Define `encode_single_image(args)` function
  - [ ] Unpack args: (image_record, config_dict)
  - [ ] Move existing encoding logic from cli_main() (lines ~427-527 in build_encodings.py) into this function
  - [ ] Load and validate image
  - [ ] Preprocess image
  - [ ] Detect faces
  - [ ] Validate single face
  - [ ] Encode face
  - [ ] Return FaceRecord or None on error
  - [ ] Ensure all errors are caught and logged

- [ ] **Modify cli_main()** in [build_encodings.py](build_encodings.py) (lines 366-563)
  - [ ] Import: `from multiprocessing import Pool, cpu_count`
  - [ ] Prepare work items: list of (image_record, config_dict) tuples
  - [ ] Calculate num_workers: `max(1, cpu_count() - 1)` (leave one core free)
  - [ ] Print: "Using N worker processes"
  - [ ] Create Pool: `with Pool(num_workers) as pool:`
  - [ ] Execute: `results = pool.map(encode_single_image, work_items)`
  - [ ] Filter None results: `new_records = [r for r in results if r is not None]`
  - [ ] Continue with existing serialization logic

- [ ] **Handle progress reporting**
  - [ ] Consider using `pool.imap()` with progress bar if needed
  - [ ] Or simple: print total completed after pool.map() finishes

#### Testing Checklist:
- [ ] Time enrollment with old code (sequential): record baseline
- [ ] Time enrollment with new code (parallel): compare
- [ ] Verify: speedup should be 4-8x on 8-core CPU
- [ ] Verify: output .pkl file is identical to sequential version
  - [ ] Use `inspect_pkl.py` to compare encodings
  - [ ] Check: same number of records
  - [ ] Check: same labels
  - [ ] Check: encoding values match (or very close due to floating point)
- [ ] Test with small dataset (10 images): verify correctness
- [ ] Test with large dataset (100+ images): verify performance gain
- [ ] Monitor CPU usage: should spike to ~90% across all cores

#### Success Criteria:
- [ ] Enrollment time reduced by 4-8x
- [ ] Output database is identical to sequential version
- [ ] No crashes or deadlocks
- [ ] CPU utilization is high during processing

---

### 2.3 Threshold Tuning Tool (2 days)

> **Scope note**: This is a one-time analysis tool, not a runtime dependency. Consider implementing as a Jupyter notebook instead of a CLI script to avoid adding matplotlib/scikit-learn as permanent project dependencies. The output (recommended threshold value) is what matters — the tool itself only needs to run once per database rebuild.

#### Tasks:
- [ ] **Create tune_threshold.py script** (new file, or Jupyter notebook)
  - [ ] Add shebang and docstring
  - [ ] Import: pickle, numpy, matplotlib, sklearn
  - [ ] Add `compute_distances(encodings, labels)` function
    - [ ] Initialize genuine_dists and impostor_dists lists
    - [ ] Double loop: compare all pairs of encodings
    - [ ] If same label: append to genuine_dists
    - [ ] If different label: append to impostor_dists
    - [ ] Return both lists
  - [ ] Add `plot_distribution(genuine, impostor, output_path)` function
    - [ ] Create figure with 2 subplots (1 row, 2 columns)
    - [ ] Subplot 1: Histogram of genuine and impostor distances
    - [ ] Use bins=50, alpha=0.7
    - [ ] Label axes and add legend
    - [ ] Subplot 2: ROC curve
    - [ ] Compute TPR, FPR, thresholds using sklearn
    - [ ] Plot ROC curve with AUC score
    - [ ] Save figure to output_path
  - [ ] Add `recommend_threshold(genuine, impostor)` function
    - [ ] Calculate EER (Equal Error Rate)
    - [ ] Calculate strict threshold: 99.9th percentile of genuine
    - [ ] Calculate lenient threshold: 0.1st percentile of impostor
    - [ ] Return dict with all three thresholds
  - [ ] Add main section
    - [ ] Parse args for database path
    - [ ] Load database
    - [ ] Compute distances
    - [ ] Print statistics: mean, std for genuine and impostor
    - [ ] Recommend thresholds
    - [ ] Print recommendations
    - [ ] Plot distribution

- [ ] **Update requirements.txt**
  - [ ] Add: `matplotlib>=3.5.0`
  - [ ] Add: `scikit-learn>=1.0.0`

#### Testing Checklist:
- [ ] Run: `python tune_threshold.py`
- [ ] Verify: no errors
- [ ] Check output: threshold_analysis.png created
- [ ] Open plot: verify histogram shows separation between genuine/impostor
- [ ] Check ROC curve: AUC should be >0.95 for good embeddings
- [ ] Verify recommendations make sense:
  - [ ] Strict threshold < Balanced threshold < Lenient threshold
  - [ ] Genuine mean << Impostor mean (good separation)
- [ ] Test with different databases
- [ ] Document recommended threshold in README

#### Success Criteria:
- [ ] Script runs without errors
- [ ] Plots are clear and informative
- [ ] Recommendations are data-driven and actionable
- [ ] User can choose threshold based on use case (strict vs lenient)

---

## Phase 3: Library Modernization - CPU Optimized (Weeks 3-4)

**Goal**: Replace Haar Cascades and dlib with modern libraries that are faster on CPU and GPU-ready for future.

### 3.1 Detection Factory with RetinaFace/MTCNN (1 week)

#### Tasks:
- [ ] **Install new dependencies**
  - [ ] Update requirements.txt: `insightface>=0.7.0`
  - [ ] Update requirements.txt: `onnxruntime>=1.15.0`
  - [ ] Update requirements.txt: `scikit-image>=0.19.0`
  - [ ] Run: `pip install insightface onnxruntime scikit-image`
  - [ ] Verify installation: `python -c "import insightface; print(insightface.__version__)"`

- [ ] **Create detector_factory.py** (new file)
  - [ ] Add imports: typing, Protocol, numpy, cv2, insightface
  - [ ] Define `FaceDetector` Protocol class
    - [ ] Method: `detect(image) -> List[Tuple[bbox, landmarks]]`
    - [ ] bbox format: (x, y, w, h)
    - [ ] landmarks format: 5x2 numpy array or None
  - [ ] Implement `HaarDetector` class
    - [ ] Init: load cascade, set parameters
    - [ ] detect(): convert to grayscale, detectMultiScale, return results
    - [ ] Return format: [(bbox, None), ...] (no landmarks)
  - [ ] Implement `RetinaFaceDetector` class
    - [ ] Init: load InsightFace model, set providers (CPU/GPU)
    - [ ] Use: `FaceAnalysis(providers=['CPUExecutionProvider'])`
    - [ ] Call: `app.prepare(ctx_id=-1)` for CPU
    - [ ] detect(): call `app.get(image)`, parse results
    - [ ] Convert bbox from [x1,y1,x2,y2] to [x,y,w,h]
    - [ ] Return format: [(bbox, landmarks), ...]
  - [ ] Implement `create_detector(detector_type, **kwargs)` factory
    - [ ] Support: "haar", "retinaface"
    - [ ] Return appropriate detector instance
  - [ ] Add `align_face(image, landmarks, output_size)` utility
    - [ ] Use scikit-image SimilarityTransform
    - [ ] Define standard reference points for 112x112 output
    - [ ] Compute transform from landmarks to reference
    - [ ] Warp image using cv2.warpAffine
    - [ ] Return aligned face image

- [ ] **Update recognize.py to use detector factory**
  - [ ] Import: `from detector_factory import create_detector, align_face`
  - [ ] Add `--detector` argument in parse_args()
    - [ ] choices=["haar", "retinaface"], default="haar"
  - [ ] Add `--align` flag in parse_args()
    - [ ] action="store_true", help="Enable face alignment"
  - [ ] Modify `detect_and_encode_faces()` function (lines 313-375)
    - [ ] Accept detector object instead of cascade_path
    - [ ] Call detector.detect(frame) instead of Haar cascade
    - [ ] Iterate through detections (bbox, landmarks)
    - [ ] If align flag and landmarks available: align face
    - [ ] Otherwise: extract ROI as before
    - [ ] Encode face and collect results
  - [ ] Update `recognize_from_webcam()` (lines 457-550)
    - [ ] Create detector: `detector = create_detector(args.detector, ...)`
    - [ ] Pass detector to detect_and_encode_faces()
  - [ ] Update `recognize_from_image()` (lines 553-611)
    - [ ] Create detector
    - [ ] Pass detector to detect_and_encode_faces()
  - [ ] Update `recognize_from_video()` (new function)
    - [ ] Create detector
    - [ ] Pass detector to detect_and_encode_faces()
  - [ ] Update `recognize_from_rtsp()` (new function)
    - [ ] Create detector
    - [ ] Pass detector to detect_and_encode_faces()

- [ ] **Update build_encodings.py to use detector factory**
  - [ ] Import: `from detector_factory import create_detector, align_face`
  - [ ] Add `--detector` argument in parse_args()
  - [ ] Add `--align` flag in parse_args()
  - [ ] Modify detect_faces() call in cli_main() (line ~437 in build_encodings.py)
    - [ ] Create detector instance
    - [ ] Call detector.detect() instead of cv2 cascade
  - [ ] Update encoding logic to use alignment if enabled

#### Testing Checklist:
- [ ] Test Haar detector (backward compatibility)
  - [ ] Run: `python recognize.py --detector haar --mode webcam`
  - [ ] Verify: works as before
- [ ] Test RetinaFace detector
  - [ ] Run: `python recognize.py --detector retinaface --mode webcam`
  - [ ] Verify: detects faces (possibly more than Haar)
  - [ ] Compare detection quality visually
- [ ] Test face alignment
  - [ ] Run: `python recognize.py --detector retinaface --align --mode image --source test.jpg`
  - [ ] Save and compare: aligned vs non-aligned faces
- [ ] Rebuild database with RetinaFace
  - [ ] Run: `python build_encodings.py --detector retinaface --align --root data/employees`
  - [ ] Compare detection rate: Haar vs RetinaFace
- [ ] Performance comparison
  - [ ] Measure FPS: Haar vs RetinaFace on CPU
  - [ ] Expect: RetinaFace slower (~10-20 FPS) but more accurate
- [ ] Side-by-side comparison
  - [ ] Process same image with both detectors
  - [ ] Save annotated outputs
  - [ ] Visually compare bounding boxes

#### Success Criteria:
- [ ] Haar detector still works (backward compatibility)
- [ ] RetinaFace provides better detection accuracy
- [ ] Face alignment improves matching quality (lower distances for genuine pairs)
- [ ] No crashes or errors with either detector
- [ ] Code is clean and maintainable with factory pattern

---

### 3.2 Embedding Factory with ArcFace (1 week)

#### Tasks:
- [ ] **Create embedding_factory.py** (new file)
  - [ ] Add imports: typing, Protocol, Optional, numpy, insightface
  - [ ] Define `FaceEmbedder` Protocol class
    - [ ] Method: `embed(face_image) -> Optional[np.ndarray]`
    - [ ] Property: `embedding_dim() -> int`
  - [ ] Implement `DlibEmbedder` class
    - [ ] Init: import face_recognition library
    - [ ] embed(): call face_recognition.face_encodings()
    - [ ] embedding_dim: return 128
  - [ ] Implement `ArcFaceEmbedder` class
    - [ ] Init: load InsightFace ArcFace model
    - [ ] Use: `get_model(model_name, providers=[...])`
    - [ ] Model options: "arcface_r50_v1", "arcface_mnet_v1"
    - [ ] Call: `model.prepare(ctx_id)` (-1 for CPU, 0+ for GPU)
    - [ ] embed(): resize face to 112x112, call model.get_feat()
    - [ ] embedding_dim: return 512
  - [ ] Implement `create_embedder(embedder_type, **kwargs)` factory
    - [ ] Support: "dlib", "arcface"
    - [ ] Return appropriate embedder instance

- [ ] **Update EncodingsDB schema** in [build_encodings.py](build_encodings.py) (lines 39-48)
  - [ ] Add field: `version: str = "schema_v2"`
  - [ ] Add field: `embedding_dim: int = 128`
  - [ ] Add field: `embedder_type: str = "dlib"`
  - [ ] Keep backward compatibility with v1 databases

- [ ] **Update database loader** in [recognize.py](recognize.py) (lines 234-267)
  - [ ] Modify `load_database()` function
  - [ ] Check for embedder_type field (use getattr with default)
  - [ ] Check for embedding_dim field (use getattr with default)
  - [ ] Print info: "Database: {embedder_type} ({embedding_dim}-D embeddings)"
  - [ ] Return: (encodings, labels, embedder_type, embedding_dim)

- [ ] **Update recognize.py to use embedding factory**
  - [ ] Import: `from embedding_factory import create_embedder`
  - [ ] Add `--embedder` argument in parse_args()
    - [ ] choices=["dlib", "arcface"], default="dlib"
  - [ ] Add `--model` argument in parse_args()
    - [ ] default="arcface_r50_v1"
    - [ ] help="ArcFace model (arcface_r50_v1 or arcface_mnet_v1)"
  - [ ] Add `--gpu` argument in parse_args()
    - [ ] type=int, default=-1
    - [ ] help="GPU device ID (-1 for CPU)"
  - [ ] In main(): load database and check embedder compatibility
    - [ ] Load: `encodings, labels, db_embedder, db_dim = load_database(...)`
    - [ ] Warn if args.embedder != db_embedder
    - [ ] Create embedder: `embedder = create_embedder(args.embedder, ...)`
  - [ ] Modify detect_and_encode_faces() to accept embedder
    - [ ] Call embedder.embed(face_roi) instead of face_recognition
    - [ ] Handle None return value

- [ ] **Update build_encodings.py to use embedding factory**
  - [ ] Import: `from embedding_factory import create_embedder`
  - [ ] Add `--embedder`, `--model`, `--gpu` arguments
  - [ ] Create embedder instance in cli_main()
  - [ ] Pass embedder to encoding functions
  - [ ] Update serialize() to save embedder_type and embedding_dim

#### Testing Checklist:
- [ ] Test dlib embedder (backward compatibility)
  - [ ] Run: `python build_encodings.py --embedder dlib`
  - [ ] Verify: database created with 128-D embeddings
- [ ] Test ArcFace embedder (CPU)
  - [ ] Run: `python build_encodings.py --embedder arcface --model arcface_mnet_v1`
  - [ ] Verify: database created with 512-D embeddings
  - [ ] Measure: encoding time per face (should be 15-20ms)
- [ ] Test ArcFace with different models
  - [ ] Test: arcface_mnet_v1 (fastest)
  - [ ] Test: arcface_r50_v1 (balanced)
  - [ ] Compare: encoding time and accuracy
- [ ] Compare embedding quality
  - [ ] Rebuild database with dlib: save as known_faces_dlib.pkl
  - [ ] Rebuild database with ArcFace: save as known_faces_arcface.pkl
  - [ ] Run tune_threshold.py on both
  - [ ] Compare: genuine/impostor separation
  - [ ] Expect: ArcFace has better separation (lower EER)
- [ ] Test recognition with ArcFace
  - [ ] Run: `python recognize.py --embedder arcface --database known_faces_arcface.pkl`
  - [ ] Verify: recognition works correctly
  - [ ] Measure: FPS improvement (should be 5-10x faster than dlib)
- [ ] Test mixed embedder warning
  - [ ] Use dlib database with arcface embedder
  - [ ] Verify: warning printed
  - [ ] Verify: matching still works (poorly)

#### Success Criteria:
- [ ] dlib embedder still works (backward compatibility)
- [ ] ArcFace provides 5-10x speedup on CPU (100ms → 10-20ms per face)
- [ ] ArcFace provides better matching accuracy
- [ ] GPU path is ready (just change ctx_id when GPU available)
- [ ] Database schema tracks embedder type for compatibility

---

## Phase 4: Multi-Camera NVR Support (Week 5)

**Goal**: Process 2-4 RTSP streams simultaneously for comprehensive coverage with centralized logging.

### 4.1 Parallel Multi-Stream Processing

#### Tasks:
- [ ] **Create recognize_multi.py** (new file)
  - [ ] Add shebang and docstring
  - [ ] Import: multiprocessing, sqlite3, datetime, json, cv2, sys
  - [ ] Import from recognize: load_database, process_frame
  - [ ] Define `init_worker(db_path, cascade_path, threshold)` function
    - [ ] Load database once per worker (global variables)
    - [ ] Print worker initialization message
  - [ ] Define `process_stream(camera_config, output_db)` function
    - [ ] Extract camera name, RTSP URL, location from config
    - [ ] Open RTSP stream with reconnection logic
    - [ ] Connect to SQLite database
    - [ ] Create detections table if not exists
    - [ ] Loop: read frames, process every 30th frame
    - [ ] For each detection: insert into database
    - [ ] Commit after each batch
    - [ ] Log progress every 10 seconds
    - [ ] Handle KeyboardInterrupt gracefully
    - [ ] Close resources in finally block
  - [ ] Define `main()` function
    - [ ] Parse arguments: config, database, cascade, threshold, output_db
    - [ ] Load camera configuration from JSON
    - [ ] Print number of cameras
    - [ ] Create multiprocessing Pool
    - [ ] Call starmap with process_stream for each camera
    - [ ] Handle KeyboardInterrupt to terminate workers

- [ ] **Create cameras.json example** (new file)
  - [ ] JSON structure with "cameras" array
  - [ ] Each camera: name, rtsp_url, location
  - [ ] Include 2-4 example cameras
  - [ ] Add comments (in separate .md file) explaining format

- [ ] **Create SQLite schema**
  - [ ] Table: detections
  - [ ] Columns: id (PRIMARY KEY), timestamp (TEXT), camera_name (TEXT), location (TEXT), identity (TEXT), confidence (REAL), distance (REAL), bbox_x (INT), bbox_y (INT), bbox_w (INT), bbox_h (INT)
  - [ ] Add index on timestamp for faster queries
  - [ ] Add index on identity for faster lookups
  - [ ] **Enable WAL mode**: `PRAGMA journal_mode=WAL` — required for safe concurrent writes from multiple processes. Without WAL, simultaneous inserts from different camera workers will cause "database is locked" errors

- [ ] **Create query examples documentation**
  - [ ] Document in README or separate QUERIES.md
  - [ ] Example 1: Recent detections (ORDER BY timestamp DESC)
  - [ ] Example 2: Unique visitors today (GROUP BY identity)
  - [ ] Example 3: Activity by camera (GROUP BY camera_name)
  - [ ] Example 4: Visitor timeline (WHERE identity=X)
  - [ ] Example 5: Hourly activity (GROUP BY hour)

#### Testing Checklist:
- [ ] Test with 1 camera
  - [ ] Run: `python recognize_multi.py --config cameras_1.json`
  - [ ] Verify: stream processes correctly
  - [ ] Check SQLite: detections inserted
  - [ ] Query: SELECT COUNT(*) FROM detections
- [ ] Test with 2 cameras
  - [ ] Run with 2 cameras in config
  - [ ] Verify: both streams process simultaneously
  - [ ] Check CPU usage: monitor with task manager
  - [ ] Verify: detections from both cameras in database
- [ ] Test with 4 cameras
  - [ ] Run with 4 cameras in config
  - [ ] Monitor: CPU usage should stay under 80%
  - [ ] Verify: all streams stable for 5+ minutes
- [ ] Test reconnection logic
  - [ ] Start with 2 cameras
  - [ ] Block one stream (unplug network)
  - [ ] Verify: other stream continues
  - [ ] Verify: blocked stream attempts reconnection
  - [ ] Restore network
  - [ ] Verify: stream reconnects successfully
- [ ] Test database queries
  - [ ] Run example queries from documentation
  - [ ] Verify: results are correct and fast
- [ ] Test long-term stability
  - [ ] Run 2-4 cameras for 30+ minutes
  - [ ] Check: no memory leaks (monitor RAM usage)
  - [ ] Check: no crashes or errors
  - [ ] Verify: database size grows steadily

#### Success Criteria:
- [ ] 2-4 cameras run simultaneously without issues
- [ ] CPU usage stays under 80% with all streams
- [ ] Reconnection logic works reliably
- [ ] SQLite logging is accurate and complete
- [ ] Queries are fast and informative
- [ ] System is stable for extended periods

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
- Phase 4: (Fill in after completion)

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

**Last Updated**: 2026-03-03
**Status**: Phase 1.1 Complete, Phase 2.1 (SimpleTracker) Complete, Phase 1.2 (RTSP) tasks complete (testing checklist pending).
**Next Action**: Complete Phase 1.2 testing checklist, then Phase 2 optimizations
