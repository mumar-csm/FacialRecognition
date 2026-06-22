# Kiosk Provisioning Runbook (HQ)

How to take a Pi from "bare OS + repo" to "boots straight into a running kiosk,
syncing to central" — with **zero keyboard/mouse at the store**. This picks up
where [`PI_SETUP.md`](PI_SETUP.md) leaves off (it explicitly defers systemd +
Chromium autostart to here).

Two layers of identity are involved, and **neither is baked into the golden
image** (see [Golden image rule](#golden-image-rule)):

- **Device identity** — `device_id` + `store_id` + the per-device `CENTRAL_API_KEY`,
  registered on central and written to `/etc/fr-kiosk/` by `provision_device.py`.
- **Tailscale node identity** — each Pi joins the tailnet with its own node key.

---

## Prerequisites

- Central tier reachable from the Pi (over the tailnet — see
  [`central/CLOUD_HOSTING.md`](central/CLOUD_HOSTING.md)). You need its base URL,
  e.g. `https://fr-central.<tailnet>.ts.net`.
- The Pi has: repo cloned at `/opt/fr-kiosk`, venv at `/opt/fr-kiosk/venv`,
  models + `data/known_faces_arcface.pkl` synced, Tailscale up. (All from
  `PI_SETUP.md` phases 1–9.)

---

## Step 1 — Register the device on central

On the central host (inside `central/`, conda env active, `.env` sourced):

```bash
make register-device DEVICE=store001-pi01 STORE=store001
```

This inserts a row in `devices` (storing only the sha256 of the key) and prints
the plaintext `CENTRAL_API_KEY` **once**. Copy it now — central can't show it
again; if lost, just re-register (rotates the key).

`DEVICE` must be globally unique; multiple Pis can share a `STORE`.

## Step 2 — Provision the Pi's config files

On the Pi (or against a staging mount during image prep), pass the key via env so
it stays out of shell history:

```bash
cd /opt/fr-kiosk
FR_PROVISION_API_KEY='<key from step 1>' sudo -E python tools/provision_device.py \
    --device-id store001-pi01 \
    --store-id store001 \
    --central-url https://fr-central.<tailnet>.ts.net \
    --enrollment-pin 1234
```

(Omit `FR_PROVISION_API_KEY` to be prompted for the key interactively instead.)

This writes:

| File | Mode | Contents |
|------|------|----------|
| `/etc/fr-kiosk/device.json` | 644 | `device_id`, `store_id`, `central_url` (+ any interval overrides) |
| `/etc/fr-kiosk/secrets.env` | 600 | `CENTRAL_API_KEY` (+ `ENROLLMENT_PIN` if `--enrollment-pin` given) |

Interval/batch flags (`--sync-interval-seconds`, `--sync-batch-size`,
`--roster-interval-seconds`) are optional — omit them and the kiosk uses its own
production defaults (1800s / 50). They're written to `device.json` only when you
override them.

**Permissions:** `secrets.env` is written mode 600 owned by whoever ran the
command (root, under `sudo`). The systemd unit runs as the Pi's login user, which
must be able to read it. Either run provisioning as that user, or
`sudo chown <user>: /etc/fr-kiosk/secrets.env` afterward.

## Step 3 — Install + enable the server unit

```bash
sudo cp /opt/fr-kiosk/deploy/fr-kiosk.service /etc/systemd/system/
sudoedit /etc/systemd/system/fr-kiosk.service   # set User= to the Pi's login user
sudo systemctl daemon-reload
sudo systemctl enable --now fr-kiosk.service
systemctl status fr-kiosk.service
journalctl -u fr-kiosk -f                        # watch it boot + workers start
```

The kiosk binds `127.0.0.1:8000` (loopback only). The sync worker and roster
client start automatically because `central_url` is set in `device.json`.

## Step 4 — Install + enable the Chromium kiosk autostart

This runs in the **graphical session** (as the desktop user, not root). The
right mechanism depends on the Pi's desktop stack:

**Option A — systemd user unit** (try this first):

```bash
chmod +x /opt/fr-kiosk/deploy/fr-kiosk-chromium.sh
mkdir -p ~/.config/systemd/user
cp /opt/fr-kiosk/deploy/fr-kiosk-chromium.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fr-kiosk-chromium.service
sudo loginctl enable-linger "$USER"   # start the user session at boot, no login
```

**Option B — XDG autostart** (use if the user unit never gets a display, common
on labwc/Wayland Bookworm):

```bash
chmod +x /opt/fr-kiosk/deploy/fr-kiosk-chromium.sh
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/fr-kiosk-chromium.desktop <<'EOF'
[Desktop Entry]
Type=Application
Name=FR Kiosk Chromium
Exec=/opt/fr-kiosk/deploy/fr-kiosk-chromium.sh
X-GNOME-Autostart-enabled=true
EOF
```

The wrapper polls `http://127.0.0.1:8000/api/health` for up to ~60s before
launching Chromium, so it tolerates the server still warming up at boot.

## Step 5 — Verify end-to-end

```bash
# server up + config correct:
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
#   -> store_id matches, enrollment_protected: true, employee_count > 0

# sync wired (after a clock event, within one sync interval):
sqlite3 data/kiosk.db \
  "SELECT event_uuid,kind,sent_at FROM outbox ORDER BY created_at DESC LIMIT 5;"
#   -> rows get sent_at populated once central acks

# reboot test — Pi should come back into the kiosk with no interaction:
sudo reboot
```

After reboot: server up (`systemctl status fr-kiosk`), Chromium full-screen on
the clock-in UI, Tailscale connected (`tailscale status`).

---

## Golden image rule

To mass-produce cards, capture a **device-agnostic** master image, then make each
clone unique at provisioning time. The image must **NOT** contain:

- `/etc/fr-kiosk/device.json` or `/etc/fr-kiosk/secrets.env` — per-device identity
  + the secret API key. Baking these in would give every Pi the same identity and
  leak one key across the whole fleet.
- The Tailscale node state (`/var/lib/tailscale/`) — every Pi needs its own node
  key. Run `sudo tailscale logout` (or wipe that dir) before capturing.

**Workflow:**

1. Build one Pi through `PI_SETUP.md` + this runbook's Steps 3–4 (unit files
   installed and enabled), but **stop before** Step 2 (don't provision identity).
2. `sudo tailscale logout`; remove `/etc/fr-kiosk/*` if present; clear shell
   history.
3. Capture the image (Pi Imager "clone" or `dd`).
4. For each new card: flash the image, boot, `tailscale up` (its own node), then
   run Steps 1–2 (`register-device` on central + `provision_device.py` on the Pi)
   and reboot.

Never commit a real `secrets.env` or a populated `device.json` to git.
