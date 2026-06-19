#!/usr/bin/env python3
"""punch.py — reusable serial bridge to the Teensy POS punch device.

Owns the serial protocol shared by the kiosk server and the bench-test CLI:
open the port, send a 7-digit POS employee ID followed by a newline, and read
back the Teensy's one-line status (``OK <id>`` or ``ERR ...``). The Teensy
(teensy_punch.ino) then emits the digits + ENTER as USB-HID keystrokes on
whichever host its USB cable is plugged into.

Used by:
    - kiosk_server.py — punches the recognized employee's ID on clock-in/out.
    - sender.py       — manual bench-test harness.

Requires
    pip install pyserial
"""

import re

import serial


POS_ID_RE = re.compile(r"^\d{7}$")

# Must match Serial.begin(...) in teensy_punch.ino.
DEFAULT_BAUD = 115200


class PosPunch:
    """A thin wrapper around a serial connection to the Teensy.

    Construction opens the port (may raise ``serial.SerialException``); callers
    that must stay alive when the Teensy is unplugged (the kiosk) should wrap the
    constructor in try/except. ``send`` does not raise for serial I/O errors on
    its own beyond what pyserial raises — the kiosk wraps ``send`` in try/except
    so a mid-shift unplug never blocks a clock event.
    """

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        # timeout=1: readline() returns after 1s if the Teensy says nothing,
        # which is plenty for it to echo its OK/ERR status line.
        self._ser = serial.Serial(port, baud, timeout=1)
        # Consume the Teensy's "teensy_punch ready" startup banner so it isn't
        # mistaken for a status line on the first send.
        self._drain()

    def _drain(self) -> None:
        """Print and consume any lines the Teensy has queued."""
        while self._ser.in_waiting:
            line = self._ser.readline().decode("ascii", errors="replace").rstrip()
            if line:
                print(f"  teensy: {line}")

    def send(self, pos_id: str) -> tuple[bool, str]:
        """Send a 7-digit ID to the Teensy and return ``(ok, detail)``.

        ``ok`` is True only when the id is well-formed and the Teensy replies
        with an ``OK`` line. A malformed id never touches the wire. ``detail``
        carries the Teensy's reply (or the reason it was refused) for logging.
        """
        if not POS_ID_RE.match(pos_id):
            return False, f"refused: {pos_id!r} is not exactly 7 digits"
        self._ser.write((pos_id + "\n").encode("ascii"))
        self._ser.flush()
        reply = self._ser.readline().decode("ascii", errors="replace").rstrip()
        return reply.startswith("OK"), reply or "no reply"

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass
