# Raspberry Pi Setup Runbook

**Goal of this session:** get a barebones working install on one Pi. End state — clock-in works once via chromium opened manually on the Pi, plus SSH access from anywhere via Tailscale.

**Explicitly NOT in this session:** systemd auto-start, chromium kiosk mode, PIN/slowapi hardening, K3.1 endpoints, Simphony integration. All of that lands later, remotely, over Tailscale SSH.

**Why this scope:** the first install is where you'll learn deployment, networking, camera setup, and ONNX-on-ARM behaviour. Don't pile new features on top of a still-unproven base. Once `git pull && restart` works over Tailscale, every subsequent change is cheap.

---

## Hardware checklist

- [ ] **Raspberry Pi 5 (8GB)** strongly recommended. Pi 4 (4GB+) works but ML inference is ~2× slower and RAM is tight.
- [ ] **Power supply** — Pi 5 needs the official 27W USB-C; Pi 4 needs ~3A USB-C. Underpowered supplies cause silent throttling and ONNX failures that look like bugs.
- [ ] **microSD card** — 32GB+ recommended (models + venv eat ~3–4GB). A20+ class for decent IO.
- [ ] **Camera** — Pi Camera Module 3 (autofocus, good low-light) preferred. USB webcam is fine and simpler in software (it's just `cv2.VideoCapture(0)`).
- [ ] **Display + HDMI cable** — needed for the camera-on-Pi test (Phase 9). Any HDMI monitor works; the long-term touchscreen can come later.
- [ ] **Keyboard + mouse** — only needed if Imager headless setup fails; otherwise SSH-only is fine.
- [ ] **Ethernet cable** — strongly recommended for the install session. WiFi works but apt + pip pull several hundred MB and ethernet is faster + more reliable.
- [ ] **SD card reader** on the Mac.

---

## Prep work on Mac (before touching the Pi)

These are 10 minutes of work that make the on-Pi session smoother.

### 1. Create `requirements-pi.txt`

Three deps in `requirements.txt` are problematic on Pi and **not used by the kiosk runtime**:

- `dlib==19.24.2` — compiles from source on ARM (~30 min), only used by the dlib embedder which the kiosk doesn't use.
- `face-recognition` — wraps dlib, same story.
- `opencv-python` — ARM wheels exist but are flaky; `opencv-python-headless` works better and we don't need GUI features (the UI is a web page).
- `matplotlib`, `scikit-learn` — only used by `tune_threshold.py`, not at runtime.

A trimmed `requirements-pi.txt` cuts install time from ~45 min to ~5 min and removes the most common failure modes.

### 2. Install Raspberry Pi Imager on Mac

Download from https://www.raspberrypi.com/software/ . This is the official tool for flashing the SD card and configuring headless boot.

### 3. Generate an SSH key pair if you don't have one

```
ls ~/.ssh/id_ed25519.pub  # if missing:
ssh-keygen -t ed25519
```

Imager can inject the public key into the Pi at flash time, so you skip password SSH entirely.

### 4. Decide on a hostname

Use something predictable like `kiosk-store-01`. This becomes the mDNS name (`kiosk-store-01.local`) for first SSH and the Tailscale name later.

---

## Phase 1: Flash OS to SD card

1. Open Raspberry Pi Imager.
2. **Choose Device:** matching your Pi.
3. **Choose OS:** Raspberry Pi OS (64-bit) — the standard one with desktop. The Lite variant works too but skip it for now since we want a desktop for the Phase 9 chromium test.
4. **Choose Storage:** the SD card.
5. **Click "Next" → "Edit Settings":**
   - **Hostname:** `kiosk-store-01`
   - **Username:** `pi` (or your preference) + strong password
   - **Configure WiFi:** SSID + password (only used if no ethernet)
   - **Locale + timezone:** match your store
   - **Services tab → Enable SSH → Allow public-key authentication only:** paste in `~/.ssh/id_ed25519.pub` from your Mac
6. Save settings, write the image (~5 min).

**Result:** SD card boots into a pre-configured Pi OS with SSH enabled and your key authorised.

---

## Phase 2: First boot + SSH

1. Insert SD card into Pi. Connect ethernet, camera, monitor (optional but useful for first boot to see what's happening), power.
2. Wait ~60 seconds for first boot.
3. From Mac:
   ```
   ssh pi@kiosk-store-01.local
   ```
4. If `.local` doesn't resolve (some routers strip mDNS), find the IP via your router's DHCP table or:
   ```
   arp -a | grep -i b8:27:eb  # Pi 4 MAC prefix
   arp -a | grep -i d8:3a:dd  # Pi 5 MAC prefix
   ```

**Verify:** you're at a `pi@kiosk-store-01:~ $` prompt without typing a password.

---

## Phase 3: System update + apt deps

```
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y \
    python3-venv python3-pip python3-dev \
    git rsync \
    libatlas-base-dev libopenblas-dev \
    libjpeg-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    v4l-utils
sudo reboot
```

`apt full-upgrade` can take 10–15 min on a fresh image. The system libraries above are what `numpy`, `Pillow`, `opencv-python-headless`, and `onnxruntime` link against at install time — installing them upfront avoids a class of pip install failures.

After reboot, SSH back in.

---

## Phase 4: Install Tailscale (do this *now*, before anything risky)

The whole point of this step happening *here* is: if Phase 5–9 break and need debugging, you want to be able to SSH in from home. Don't leave the store without Tailscale working.

```
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

The CLI prints a URL — open it on your Mac, log in to your Tailscale account (create one if needed; free for up to 100 devices).

After auth completes:
```
tailscale ip -4   # note this IP
tailscale status  # confirm "logged out" became "active"
```

**Verify from Mac (still on store WiFi):**
```
ssh pi@<tailscale-name-or-ip>
```

**Now verify it works *off* the store network:** tether your Mac to your phone's cellular, then:
```
ssh pi@<tailscale-name-or-ip>
```
Should still connect. **If this fails, fix it before proceeding** — this is your lifeline for everything after this session.

---

## Phase 5: Clone repo + Python venv

```
sudo mkdir -p /opt/fr-kiosk
sudo chown pi:pi /opt/fr-kiosk
git clone <your-repo-url> /opt/fr-kiosk
cd /opt/fr-kiosk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

Pip install will take ~5 min. `onnxruntime` and `insightface` are the slowest. Watch for:
- `onnxruntime` should install a prebuilt aarch64 wheel — if it tries to build from source, something's wrong with the Python version (need 3.9+).
- `insightface` pulls in `numpy`, which is why we installed `libopenblas-dev` upfront.

**Verify:**
```
python -c "import cv2, numpy, onnxruntime, insightface, fastapi; print('ok')"
```

---

## Phase 6: Sync models + encodings from Mac

`models/` and `data/*.pkl` are gitignored. Get them onto the Pi via rsync from the Mac:

```
# Run from your Mac, not the Pi
rsync -av \
    /Users/muhammedumar/Desktop/FR/FacialRecognition/models/ \
    pi@kiosk-store-01.local:/opt/fr-kiosk/models/

rsync -av \
    /Users/muhammedumar/Desktop/FR/FacialRecognition/data/known_faces_arcface.pkl \
    pi@kiosk-store-01.local:/opt/fr-kiosk/data/
```

(`buffalo_l/` for InsightFace is downloaded automatically the first time it's used, so don't worry about that one.)

**Verify on Pi:**
```
ls -lh models/anti_spoof.onnx           # ~612KB
ls -lh data/known_faces_arcface.pkl
```

---

## Phase 7: Camera sanity check

### If Pi Camera Module:
```
libcamera-hello --timeout 3000   # 3-second preview window (needs monitor)
libcamera-still -o /tmp/test.jpg # capture a still
ls -lh /tmp/test.jpg
```

If `libcamera-*` commands work, the camera is detected at the system level. Now check OpenCV sees it:
```
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('camera:', 'OK' if ret else 'FAILED', frame.shape if ret else '')
cap.release()
"
```

If `cv2.VideoCapture(0)` fails on Pi Camera Module, you may need to enable the libcamera v4l2 shim (`sudo raspi-config` → Interface Options → Camera, or set `dtoverlay=imx708` in `/boot/firmware/config.txt`). USB webcam users skip this entirely.

### If USB webcam:
```
ls /dev/video*
v4l2-ctl --list-devices
```
Then run the OpenCV check above. Should just work.

---

## Phase 8: First server run + smoke test (via SSH tunnel from Mac)

The server binds to `127.0.0.1` (loopback only) per recent hardening. To hit it from the Mac for testing, use an SSH tunnel — your browser on the Mac talks to a local port, SSH forwards to the Pi's loopback.

**On the Pi:**
```
cd /opt/fr-kiosk
source venv/bin/activate
python kiosk_server.py \
    --database data/known_faces_arcface.pkl \
    --enrollment-pin "1234"
```

Expected output: model load logs, then `Uvicorn running on http://127.0.0.1:8000`.

**On the Mac (new terminal):**
```
ssh -L 8000:127.0.0.1:8000 pi@<tailscale-name>
# leave that session open — it's the tunnel
```

**On the Mac (browser):** http://localhost:8000

The kiosk page should load, your **Mac's webcam** activates (the camera capture happens in the browser, not on the Pi), and you can attempt a clock-in. This proves the **server pipeline works on the Pi** — model load, detection, anti-spoof, embedding, matching, SQLite write — without depending on the Pi's camera.

**Verify clock-in landed:**
```
sqlite3 /opt/fr-kiosk/data/kiosk.db "SELECT * FROM attendance ORDER BY id DESC LIMIT 5;"
```

---

## Phase 9: Camera-on-Pi test (chromium on the Pi)

Phase 8 proved the server runs. This phase proves the Pi's camera works in chromium's `getUserMedia` — the actual production path.

**On the Pi (with monitor + keyboard, or via VNC):**

1. Make sure `kiosk_server.py` is still running (or start it again).
2. Launch chromium:
   ```
   chromium-browser http://127.0.0.1:8000
   ```
3. Grant camera permission when prompted.
4. Confirm: live camera feed shows, you can clock in using the Pi's camera.

**If chromium can't see the camera but `cv2.VideoCapture(0)` worked in Phase 7:** there's a `getUserMedia`-specific permission or v4l2 issue. Most likely fix: ensure your user is in the `video` group:
```
sudo usermod -aG video pi
# log out and back in
```

---

## Phase 10: Wrap up before leaving the store

- [ ] Tailscale SSH from off-network confirmed working (cellular hotspot test).
- [ ] Server runs, clock-in records in SQLite.
- [ ] Pi's camera works in chromium.
- [ ] Pi survives a `sudo reboot` and you can SSH in afterwards.
- [ ] Note the Tailscale hostname and store it somewhere you'll remember.

The Pi can now be left running. You don't need to start `kiosk_server.py` automatically yet — that's the systemd unit we'll add in the next session, remotely.

---

## Common pitfalls

- **`pip install` of opencv/onnxruntime hangs forever** — usually means it's compiling from source instead of using a wheel. Check Python version (`python --version` should be 3.9+) and architecture (`uname -m` should be `aarch64`, not `armv7l` — if armv7l, you flashed the 32-bit OS by mistake).
- **`Could not open camera`** in Phase 7 with Pi Camera Module — the libcamera v4l2 shim isn't loaded. `sudo raspi-config` → enable camera, or add `dtoverlay=imx708` (Camera Module 3) / `dtoverlay=imx219` (Camera Module 2) to `/boot/firmware/config.txt`.
- **Model load takes 30+ seconds** — normal on first run; InsightFace downloads `buffalo_l/` from the internet (~280MB) and caches under `~/.insightface/models/`. Subsequent runs load in 2–3 sec.
- **Clock-in works once then never again** — cooldown is 120s by default. Either wait or pass `--cooldown 0` for testing.
- **Pi runs hot under inference** — sustained ML on Pi 5 is ~75°C without a heatsink. A passive heatsink or active cooler is worth it for a 24/7 kiosk. Without it, thermal throttling kicks in and FPS halves.
- **SD card wear** — SQLite writes constantly. For long-term deployment, an SSD over USB 3 is dramatically more reliable than an SD card. Out of scope for this session; flag it for the production build.

---

## What comes after this session (remote, over Tailscale)

These are the follow-up sessions, each independently doable from home:

1. **K3.1 finish + hardening** — DELETE / GET employees endpoints, PIN length bump, slowapi rate limiting, inactivity timeout, audit log.
2. **systemd + chromium kiosk autostart** — `kiosk_server.service` unit, chromium kiosk mode launched on desktop login, Pi boots straight into the kiosk UI.
3. **Simphony API discovery + integration** — turns the audit-only kiosk into a real POS gate.
4. **30-store rollout tooling** — GitHub Actions over Tailscale, per-store config templating.

Each of those is a separate plan when the time comes.