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
import sys

import serial

from punch import DEFAULT_BAUD, PosPunch


def send_one(punch: PosPunch, pos_id: str) -> None:
    ok, detail = punch.send(pos_id)
    # send() refuses malformed IDs without touching the wire; surface that to
    # stderr. A real Teensy reply (OK/ERR) goes to stdout for the operator.
    if not ok and detail.startswith("refused"):
        print(f"  {detail}", file=sys.stderr)
    else:
        print(f"  teensy: {detail}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", required=True,
                   help="serial device path (e.g. /dev/cu.usbmodem12345)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                   help=f"serial baud (must match the Teensy sketch; default {DEFAULT_BAUD})")
    p.add_argument("--id", default=None,
                   help="if set, send this ID and exit; otherwise interactive")
    args = p.parse_args()

    try:
        punch = PosPunch(args.port, args.baud)
    except serial.SerialException as e:
        print(f"could not open {args.port}: {e}", file=sys.stderr)
        return 1

    if args.id is not None:
        send_one(punch, args.id)
    else:
        print("ready — enter 7-digit IDs, Ctrl-C to exit")
        try:
            while True:
                pos_id = input("> ").strip()
                if not pos_id:
                    continue
                send_one(punch, pos_id)
        except (KeyboardInterrupt, EOFError):
            print()

    punch.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
