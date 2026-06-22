# pos_bridge

Hardware bridge that types a recognized employee's 7-digit POS ID into the
Oracle POS terminal, posing as a USB keyboard. Lets face recognition replace
the customer's existing fingerprint or manual-digit punch flow **without**
integrating any Oracle-side API.

The kiosk wiring is in place: `kiosk_server.py` imports `PosPunch` from
`punch.py` and, on every successful clock-in/out, sends the recognized
employee's POS ID to the Teensy. Enable it by pointing the server at the serial
device:

```bash
python kiosk_server.py --database data/known_faces_arcface.pkl \
  --pos-serial-port /dev/cu.usbmodem12345   # add --pos-baud if not 115200
```

Punching is **best-effort** — attendance is always recorded even if the Teensy
is unplugged or the port is wrong (those just log a warning), and the feature is
off entirely when `--pos-serial-port` is omitted. Run the bench test below first
to confirm the hardware chain before relying on it in the store.

## Architecture

```
  ┌───────────┐   serial   ┌────────┐    USB HID    ┌──────────────────┐
  │  Pi (or   │ ─────────► │ Teensy │ ────────────► │  Oracle POS      │
  │   Mac for │  7 digits  │        │  keystrokes   │  terminal        │
  │   bench)  │   + "\n"   │        │   + ENTER     │  (clock-in form) │
  └───────────┘            └────────┘               └──────────────────┘
```

The Teensy reads a digit string over serial, validates it, then emits the
digits + ENTER as if a USB keyboard had typed them on whichever host the
Teensy's USB cable is plugged into.

## Files

- `teensy_punch/teensy_punch.ino` — Arduino sketch (Teensyduino) that reads the
  ID from **USB serial**. The folder name must match the sketch filename — that's
  how the Arduino IDE expects sketch projects to be laid out.
- `teensy_punch_uart/teensy_punch_uart.ino` — same sketch but reads from the
  **`Serial1` UART pins** instead of USB, for the Pi-on-pins → POS-on-USB
  deployment. See **Firmware variants** below.
- `punch.py` — reusable `PosPunch` class that owns the serial protocol (open,
  send a 7-digit ID, read back the Teensy's `OK`/`ERR` reply). Imported by both
  `kiosk_server.py` (live punching) and `sender.py` (bench testing).
- `sender.py` — Python harness that sends test IDs to the Teensy over serial,
  built on `punch.py`. Used for bench testing; not part of the kiosk runtime.

## Firmware variants

There are two sketches. They speak the **identical wire protocol** (send 7 ASCII
digits + newline, get back `OK <id>` / `ERR ...`, keystrokes emitted over USB) —
the only difference is which port the Teensy *reads the ID from*. Flash the one
that matches your topology; nothing on the Python/kiosk side changes either way.

| Variant | Reads ID from | Topology | Kiosk `--pos-serial-port` |
|---|---|---|---|
| `teensy_punch` (USB) | USB `Serial` | **single host** — the machine the Teensy's USB plugs into is both sender *and* keystroke receiver (Pi-to-Pi bench test) | `/dev/ttyACM0` (Linux) / `/dev/cu.usbmodem*` (macOS) |
| `teensy_punch_uart` (UART) | `Serial1` pins | **two hosts** — Pi sends over its GPIO UART, Teensy types into a *separate* POS over USB (production) | `/dev/serial0` (the Pi's UART) |

Why two: the Teensy has one USB port, and in "Serial + Keyboard" mode that single
cable carries both the serial link and the keyboard *to the same host*. So USB
serial can't drive a sender ≠ receiver split — that needs the hardware UART pins.

### UART wiring (Pi ↔ Teensy 4.x)

Pi GPIO and Teensy 4.x pins are both **3.3V**, so they connect directly — **no
level shifter**. Cross TX↔RX, and share a ground:

| Raspberry Pi | → | Teensy 4.x |
|---|---|---|
| TXD0 — GPIO14 (physical pin 8) | → | RX1 — pin 0 |
| RXD0 — GPIO15 (physical pin 10) | → | TX1 — pin 1 |
| GND (physical pin 6) | → | GND |

> Teensy 4.x pins are **not** 5V-tolerant — fine here since the Pi is 3.3V, just
> never feed them 5V.

### Enable the Pi's UART (one-time)

By default the Pi uses the UART for a serial login console. Free it up:

```bash
sudo raspi-config
#   Interface Options → Serial Port
#     "login shell accessible over serial?"  → No
#     "serial port hardware enabled?"         → Yes
sudo reboot
```

The UART is then `/dev/serial0` (a stable symlink to the primary UART). Sanity-
check the UART itself **before** involving the Teensy by jumpering GPIO14↔GPIO15
and running the probe below — it should echo your own bytes back:

```bash
python3 -c "import serial,time; s=serial.Serial('/dev/serial0',115200,timeout=2); time.sleep(1); s.write(b'1234567\n'); print(s.readline())"
# loopback jumper present → b'1234567\n'   |   Teensy + UART firmware → b'OK 1234567\r\n'
```

Then start the kiosk against it: `python kiosk_server.py ... --pos-serial-port /dev/serial0`.

## Bench test (Mac, no POS terminal needed)

This uses the **`teensy_punch` (USB)** variant. It proves the entire hardware
chain — Pi/Mac → Teensy → keystrokes-into-app — works in isolation, using your
Mac in place of both the Pi (sender) and the POS terminal (keystroke receiver).

1. **Install the Arduino IDE** from <https://www.arduino.cc/en/software>.
2. **Install Teensyduino** from <https://www.pjrc.com/teensy/teensyduino.html>.
   This is the PJRC plugin that teaches the Arduino IDE about Teensy boards.
3. **Plug the Teensy into your Mac via USB.**
4. **Open the sketch** in the Arduino IDE: File → Open → `teensy_punch/teensy_punch.ino`.
5. **Configure the board** in the Tools menu:
   - Board: select your Teensy model (e.g. Teensy 4.0, Teensy 4.1, Teensy LC)
   - **USB Type: "Serial + Keyboard"** ← easy to miss; without this the Teensy
     can't emit keystrokes
6. **Click Upload.** The IDE compiles the sketch and flashes it to the Teensy.
7. **Find the serial port** that the Teensy enumerated as:
   ```bash
   ls /dev/cu.usbmodem*    # macOS
   ls /dev/ttyACM*         # Linux / Raspberry Pi
   ```
8. **Install pyserial** (one-time) in the conda env:
   ```bash
   conda run -n face_recognition_env pip install pyserial
   ```
9. **Open TextEdit** (or any text editor) and click into the document so it
   has keyboard focus. This is the stand-in for the Oracle POS clock-in form.
10. **Send a test punch:**
    ```bash
    conda run --no-capture-output -n face_recognition_env python sender.py \
        --port /dev/cu.usbmodemXXXXX --id 1234567
    ```
11. **Expected:**
    - The terminal prints `teensy: teensy_punch ready` then `teensy: OK 1234567`.
    - `1234567` and a newline appear in TextEdit.

If both happen, the chain is proven and we can move on to wiring the kiosk.

## Bench test (Raspberry Pi, from scratch)

Same proof as the Mac test — Pi → Teensy → keystrokes-into-an-app — but run
entirely on the Pi, with the Pi standing in for both the sender and the POS
terminal. This assumes the Teensy has **not** been flashed yet, so it covers
the whole chain end-to-end without ever touching a Mac. Do it at the Pi with a
monitor + keyboard (the desktop is needed to flash and to receive keystrokes).

### 1. Install the Arduino IDE + Teensyduino on the Pi

Raspberry Pi OS (64-bit) is ARM64 — confirm with `uname -m` (should print
`aarch64`).

> ⚠️ **Don't use the Arduino IDE 2.x AppImage/zip on the main download page** —
> those are X86-64 (Intel/AMD) and won't run on the Pi. Arduino does **not**
> ship an official ARM64 build of IDE 2.x. The supported path on Raspberry Pi
> OS 64-bit is the **legacy Arduino IDE 1.8.19** (which has an ARM64 build)
> plus the Teensyduino AArch64 installer.

1. Download **Arduino IDE 1.8.19 — "Linux ARM 64-bit"** from the
   legacy/previous-releases section of <https://www.arduino.cc/en/software>.
   It's a `.tar.xz`; extract it (e.g. to `~/arduino-1.8.19`):
   ```bash
   tar -xf ~/Downloads/arduino-1.8.19-linuxaarch64.tar.xz -C ~
   ```
2. Download the Teensyduino installer **`TeensyduinoInstall.linuxaarch64`**
   (the "Linux Installer — ARM 64 bit / AARCH64" entry) from
   <https://www.pjrc.com/teensy/td_download.html>, then run it and point it at
   the extracted Arduino folder when prompted:
   ```bash
   chmod 755 TeensyduinoInstall.linuxaarch64
   ./TeensyduinoInstall.linuxaarch64
   ```
3. Install the **PJRC udev rules** so a normal user can flash the board
   without root (the installer/page provides `00-teensy.rules`):
   ```bash
   sudo cp 00-teensy.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   ```
   Unplug and replug the Teensy after this so the new rules apply.

### 2. Flash the sketch

1. **Plug the Teensy into the Pi via USB.**
2. In the Arduino IDE: File → Open → `teensy_punch/teensy_punch.ino`.
3. Tools menu:
   - Board: your Teensy model (e.g. Teensy 4.0 / 4.1 / LC)
   - **USB Type: "Serial + Keyboard"** ← required, or the Pi won't see HID
     keystrokes
4. Click **Upload**. The IDE compiles and flashes the Teensy.

### 3. Install pyserial in the Pi venv

The Pi uses a venv (not conda — that's Mac-only). From the repo root:
```bash
source venv/bin/activate
pip install pyserial
```

### 4. Find the serial port

On Linux the Teensy enumerates under `/dev/ttyACM*`:
```bash
ls /dev/ttyACM*    # usually /dev/ttyACM0
```

### 5. Grant serial permission (one-time)

Opening the port as a normal user fails with "permission denied" unless you're
in the `dialout` group:
```bash
sudo usermod -aG dialout $USER
```
Group membership is only read at login, so **reboot** (`sudo reboot`) or log
out and back in for it to take effect.

### 6. Send a test punch and observe it

The Teensy types like a USB keyboard: `sender.py` only ships the digits *to the
Teensy over the wire*; the Teensy then "types" them into **whichever window has
focus on the Pi's screen**. The simplest proof uses one-shot mode with the
terminal focused:

```bash
python pos_bridge/sender.py --port /dev/ttyACM0 --id 1234567
```

**Expected:**
- The terminal prints `teensy: teensy_punch ready` then `teensy: OK 1234567`.
- A moment later the Teensy types `1234567` at your shell prompt, so bash
  responds `1234567: command not found`. **That phantom command is the proof**
  — those digits were injected as HID keystrokes, not typed by you.

> ⚠️ **Don't use interactive mode with the terminal focused.** In interactive
> mode (`sender.py` without `--id`) the Teensy's injected digits land back in
> the sender's own `>` prompt, which re-sends them and loops. For interactive
> testing, open a **separate text editor** (Text Editor / Mousepad on the Pi
> desktop), click into it so *it* has focus, and the keystrokes go there
> instead — run the validation cases below that way.

## Validation cases worth running

After the basic test works, send these via `sender.py` interactive mode to
make sure the Teensy rejects bad input cleanly:

| Input        | Expected Teensy response       | Should it type? |
|--------------|--------------------------------|-----------------|
| `1234567`    | `OK 1234567`                   | yes             |
| `123456`     | `ERR bad_length 6`             | no              |
| `12345678`   | `ERR bad_length 8`             | no              |
| `abc4567`    | `ERR non_digit`                | no              |
| (empty line) | (silent — buffer was empty)    | no              |

## Production deployment (deferred until bench test passes)

The bench-test wiring (Teensy USB → Mac) won't work in production because
the Teensy's USB cable will be plugged into the **POS terminal**, not the Pi
— that's how the POS sees the keystrokes. So the Pi needs a different way
to talk to the Teensy:

- **Teensy USB** → POS terminal (carries HID keystrokes)
- **Teensy hardware UART pins (TX/RX/GND)** → Pi GPIO header (carries the
  digit string from the Pi)

The current sketch reads from `Serial` (USB CDC). For production we'll add a
read from `Serial1` (the hardware UART), so the same sketch works for both
bench tests and the real wiring. Defer until the bench test confirms HID
itself works.

## Can we test without the Teensy?

Mostly no, and it would be wheel-spinning. The whole risk this prototype
exists to retire is "does USB HID injection actually land in the focused
field of another application." That's a hardware behavior, not a software
contract — no mock can answer it.

What *can* be verified without hardware:

- `sender.py` syntax (already confirmed)
- The Teensy sketch will compile in the Arduino IDE with the Verify button
  (catches typos, doesn't prove behavior)

What can't be verified without hardware:

- HID keystroke emission into another app
- Whether the Pi → Teensy serial link is electrically sound
- Whether the POS terminal accepts the input (requires a store visit)

**Recommended:** wait for the Teensy. Use the time on Pi tuning or starting
the OHIP-credentials conversation in parallel as a fallback option.

## Why this approach over Oracle's OHIP API

Quick reminder for future readers: we considered a path where the Pi calls
Oracle's Hospitality Integration Platform (OHIP) REST API directly to record
punches. That path was sized at ~1 week of engineering once credentials and
API docs were in hand from Oracle — but the credentials themselves typically
take weeks of vendor back-and-forth to obtain, with no certainty about which
Simphony modules the customer has licensed. The Teensy approach removes
Oracle as an external vendor dependency entirely, which is the real win.
The OHIP option remains a viable fallback if HID injection runs into
problems at the first store visit.
