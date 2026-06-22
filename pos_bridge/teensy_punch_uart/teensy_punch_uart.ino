// teensy_punch_uart.ino — emit a 7-digit POS employee ID as USB HID keystrokes.
//
// UART variant of teensy_punch.ino. Reads the incoming ID from Serial1 (the
// hardware UART on Teensy pins 0=RX1, 1=TX1) instead of USB serial, so the
// sender (the Pi, over its GPIO UART) and the keystroke receiver (the POS, over
// USB) can be two DIFFERENT machines:
//
//   Pi (kiosk) --UART pins--> Teensy --USB HID--> POS terminal
//
// The wire protocol is identical to the USB variant: buffer characters until a
// newline, validate exactly 7 ASCII digits, then "type" those digits + ENTER on
// the USB host as a keyboard would. Status replies (OK/ERR) go back over Serial1
// so the Pi can read them.
//
// Board:  Teensy 4.0 / 4.1 / LC (any with USB HID + a hardware UART)
// IDE:    Arduino IDE with the Teensyduino add-on installed
// Setup:  Tools > USB Type must be set to "Serial + Keyboard" (the USB Keyboard
//         is what types into the POS; the USB serial goes unused here). Flashing
//         works regardless of USB Type — it uses the HalfKay bootloader/button.
// Wiring: Pi TXD0 GPIO14 (phys pin 8) -> Teensy RX1 (pin 0)
//         Pi RXD0 GPIO15 (phys pin 10) -> Teensy TX1 (pin 1)   (cross TX<->RX)
//         Pi GND (phys pin 6)          -> Teensy GND
//         Both run at 3.3V — no level shifter needed. Common ground is required.
//         Enable the Pi UART (raspi-config: serial login shell off, hardware on);
//         the kiosk then uses --pos-serial-port /dev/serial0.

const size_t MAX_BUF = 16;   // 7 expected + slack so junk on the wire can't lock us up
char    buf[MAX_BUF];
size_t  buf_len = 0;

void setup() {
  Serial1.begin(115200);
  // Brief settle before we banner.
  delay(200);
  Serial1.println("teensy_punch ready");
}

void loop() {
  while (Serial1.available() > 0) {
    char c = (char)Serial1.read();

    if (c == '\n' || c == '\r') {
      process_buffer();
      buf_len = 0;
    } else if (buf_len < MAX_BUF - 1) {
      buf[buf_len++] = c;
    } else {
      // Overflow — somebody is sending garbage. Reset, warn, keep running.
      buf_len = 0;
      Serial1.println("ERR overflow");
    }
  }
}

void process_buffer() {
  if (buf_len == 0) return;   // empty line, ignore

  if (buf_len != 7) {
    Serial1.print("ERR bad_length ");
    Serial1.println(buf_len);
    return;
  }
  for (size_t i = 0; i < 7; i++) {
    if (buf[i] < '0' || buf[i] > '9') {
      Serial1.println("ERR non_digit");
      return;
    }
  }

  // Null-terminate so Keyboard.println treats the buffer as a C string,
  // then emit the 7 digits + ENTER as USB keystrokes on the host (the POS).
  char out[8];
  for (size_t i = 0; i < 7; i++) out[i] = buf[i];
  out[7] = '\0';
  Keyboard.println(out);

  Serial1.print("OK ");
  Serial1.println(out);
}
