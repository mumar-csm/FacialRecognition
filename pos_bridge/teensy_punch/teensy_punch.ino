// teensy_punch.ino — emit a 7-digit POS employee ID as USB HID keystrokes.
//
// Reads from USB serial at 115200 baud. Buffers characters until a newline
// arrives, then validates the buffer is exactly 7 ASCII digits. If valid,
// "types" those digits + ENTER on the host as a USB keyboard would. If
// invalid, discards the buffer and emits a brief error back over serial so
// the sender can see what went wrong during bench-testing.
//
// Board:  Teensy 4.0 / 4.1 / LC (any with USB HID + Serial)
// IDE:    Arduino IDE with the Teensyduino add-on installed
// Setup:  Tools > USB Type must be set to "Serial + Keyboard"
//         (without this the host won't see HID keystrokes)

const size_t MAX_BUF = 16;   // 7 expected + slack so junk on the wire can't lock us up
char    buf[MAX_BUF];
size_t  buf_len = 0;

void setup() {
  Serial.begin(115200);
  // Brief settle so the host has enumerated us before we banner.
  delay(200);
  Serial.println("teensy_punch ready");
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      process_buffer();
      buf_len = 0;
    } else if (buf_len < MAX_BUF - 1) {
      buf[buf_len++] = c;
    } else {
      // Overflow — somebody is sending garbage. Reset, warn, keep running.
      buf_len = 0;
      Serial.println("ERR overflow");
    }
  }
}

void process_buffer() {
  if (buf_len == 0) return;   // empty line, ignore

  if (buf_len != 7) {
    Serial.print("ERR bad_length ");
    Serial.println(buf_len);
    return;
  }
  for (size_t i = 0; i < 7; i++) {
    if (buf[i] < '0' || buf[i] > '9') {
      Serial.println("ERR non_digit");
      return;
    }
  }

  // Null-terminate so Keyboard.println treats the buffer as a C string,
  // then emit the 7 digits + ENTER as USB keystrokes on the host.
  char out[8];
  for (size_t i = 0; i < 7; i++) out[i] = buf[i];
  out[7] = '\0';
  Keyboard.println(out);

  Serial.print("OK ");
  Serial.println(out);
}
