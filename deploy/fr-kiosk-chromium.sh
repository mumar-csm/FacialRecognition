#!/usr/bin/env bash
# fr-kiosk-chromium.sh — wait for the kiosk API to come up, then launch Chromium
# full-screen in kiosk mode pointed at the local server.
#
# Run from the graphical session (see fr-kiosk-chromium.service or the XDG
# autostart entry in PROVISIONING.md). fr-kiosk.service must be running.
set -euo pipefail

URL="${FR_KIOSK_URL:-http://127.0.0.1:8000}"
HEALTH="${URL}/api/health"

# Chromium ships as 'chromium-browser' on older Raspberry Pi OS and 'chromium'
# on Bookworm. Use whichever exists.
BROWSER="$(command -v chromium-browser || command -v chromium || true)"
if [ -z "${BROWSER}" ]; then
    echo "fr-kiosk-chromium: no chromium/chromium-browser on PATH" >&2
    exit 1
fi

# Poll /api/health until the server answers (bounded: ~60s).
for _ in $(seq 1 60); do
    if curl -fs "${HEALTH}" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

exec "${BROWSER}" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-pinch \
    --incognito \
    "${URL}"
