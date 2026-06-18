#!/usr/bin/env python3
"""sender.py — bench-test harness for the Teensy POS punch bridge.

Sends a 7-digit POS employee ID over USB serial to a Teensy running
teensy_punch.ino. Pair this with a text editor open and focused on the
host machine to verify the Teensy emits the digits as USB HID keystrokes
(the digits + a newline should appear in the text editor).

Examples
    # One-shot send and exit
    python sender.py --port /dev/cu.usbmodem12345 --id 1234567

    # Interactive — prompts for IDs until Ctrl-C
    python sender.py --port /dev/cu.usbmodem12345

Finding the serial port
    macOS:  ls /dev/cu.usbmodem*   (Teensy enumerates as a usbmodem device)
    Linux:  ls /dev/ttyACM*        (or /dev/ttyUSB* on some setups)

Requires
    pip install pyserial
"""

import argparse
import re
import sys

import serial


POS_ID_RE = re.compile(r"^\d{7}$")


def drain(ser: serial.Serial) -> None:
    """Print and consume any lines the Teensy has queued."""
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="replace").rstrip()
        if line:
            print(f"  teensy: {line}")


def send_one(ser: serial.Serial, pos_id: str) -> None:
    if not POS_ID_RE.match(pos_id):
        print(f"  refusing to send: {pos_id!r} is not exactly 7 digits",
              file=sys.stderr)
        return
    ser.write((pos_id + "\n").encode("ascii"))
    ser.flush()
    # Read the Teensy's one-line response (OK ... or ERR ...). timeout=1 on
    # the serial open means readline returns after 1s if nothing arrives,
    # which is plenty for the Teensy to echo a status line.
    line = ser.readline().decode("ascii", errors="replace").rstrip()
    if line:
        print(f"  teensy: {line}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", required=True,
                   help="serial device path (e.g. /dev/cu.usbmodem12345)")
    p.add_argument("--baud", type=int, default=115200,
                   help="serial baud (must match the Teensy sketch; default 115200)")
    p.add_argument("--id", default=None,
                   help="if set, send this ID and exit; otherwise interactive")
    args = p.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"could not open {args.port}: {e}", file=sys.stderr)
        return 1

    # Catch the Teensy's startup banner so we know it's enumerated.
    drain(ser)

    if args.id is not None:
        send_one(ser, args.id)
    else:
        print("ready — enter 7-digit IDs, Ctrl-C to exit")
        try:
            while True:
                pos_id = input("> ").strip()
                if not pos_id:
                    continue
                send_one(ser, pos_id)
        except (KeyboardInterrupt, EOFError):
            print()

    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
