#!/usr/bin/env bash
# lock_camera.sh — pin the kiosk webcam's exposure / gain / white-balance to
# fixed, calibrated values so auto-exposure can't over-brighten the face crop
# (the root cause of MiniFAS false "spoof" rejections — see PROD_HARDENING.md /
# the Stage 0 plan).
#
# Browser-agnostic: this talks to the V4L2 device directly, so it works whether
# the kiosk runs Firefox or Chromium. Because a browser can RESET these controls
# when it (re)opens the camera, run this AFTER the browser has the stream open,
# and/or re-apply on a timer (see fr-camera-lock.service).
#
# Calibration values come from /etc/fr-kiosk/camera.env (see calibrate_camera.sh).
# Usage:
#   sudo ./lock_camera.sh                 # uses /etc/fr-kiosk/camera.env
#   sudo CAM_ENV=/path/to.env ./lock_camera.sh
#   sudo FR_CAM_DEVICE=/dev/video0 FR_CAM_EXPOSURE=156 ./lock_camera.sh
set -euo pipefail

CAM_ENV="${CAM_ENV:-/etc/fr-kiosk/camera.env}"
# shellcheck disable=SC1090
[ -f "$CAM_ENV" ] && . "$CAM_ENV"

DEV="${FR_CAM_DEVICE:-}"
EXPOSURE="${FR_CAM_EXPOSURE:-}"
GAIN="${FR_CAM_GAIN:-}"
WB="${FR_CAM_WB:-}"

command -v v4l2-ctl >/dev/null 2>&1 || {
    echo "lock_camera: v4l2-ctl not found — install with: sudo apt install v4l-utils" >&2
    exit 1
}

# ── Pick the UVC (USB) capture node ──────────────────────────────────────────
# Skip CSI/libcamera nodes (bcm2835, unicam, pispbe) — we want the USB webcam.
if [ -z "$DEV" ]; then
    for d in /dev/video*; do
        [ -e "$d" ] || continue
        drv="$(v4l2-ctl -d "$d" -D 2>/dev/null | awk -F': *' '/Driver name/{print $2}')"
        caps="$(v4l2-ctl -d "$d" --list-ctrls 2>/dev/null || true)"
        # A real capture node with exposure controls is what we want.
        if echo "$caps" | grep -qiE 'exposure'; then
            case "$drv" in
                *bcm2835*|*unicam*|*pispbe*) continue ;;
            esac
            DEV="$d"
            break
        fi
    done
fi
[ -n "$DEV" ] || { echo "lock_camera: could not find a webcam node with exposure controls" >&2; exit 1; }
echo "lock_camera: using device $DEV"

CTRLS="$(v4l2-ctl -d "$DEV" --list-ctrls 2>/dev/null || true)"

# Resolve a control name across old/new kernel spellings. Returns the first that
# actually exists on this camera, or empty.
resolve() {
    for name in "$@"; do
        if echo "$CTRLS" | grep -qE "^[[:space:]]*${name}[[:space:]]"; then
            echo "$name"; return 0
        fi
    done
    echo ""
}

set_ctrl() {  # set_ctrl <name> <value>
    local name="$1" val="$2"
    [ -n "$name" ] && [ -n "$val" ] || return 0
    if v4l2-ctl -d "$DEV" --set-ctrl "${name}=${val}" 2>/dev/null; then
        echo "  set ${name} = ${val}"
    else
        echo "  WARN: failed to set ${name}=${val} (out of range or read-only?)" >&2
    fi
}

# Logical control -> possible real names (new bookworm first, then legacy).
AE_CTRL="$(resolve auto_exposure exposure_auto)"
EXP_CTRL="$(resolve exposure_time_absolute exposure_absolute)"
GAIN_CTRL="$(resolve gain)"
WBA_CTRL="$(resolve white_balance_automatic white_balance_temperature_auto)"
WB_CTRL="$(resolve white_balance_temperature)"
BLC_CTRL="$(resolve backlight_compensation)"

echo "lock_camera: applying manual exposure/gain/white-balance"

# 1) Manual exposure mode. UVC convention: 1 = Manual, 3 = Aperture Priority(auto).
set_ctrl "$AE_CTRL" 1
# 2) Fixed exposure time + gain (calibrated).
set_ctrl "$EXP_CTRL" "$EXPOSURE"
set_ctrl "$GAIN_CTRL" "$GAIN"
# 3) Manual white balance (0 = off/manual) + fixed temperature.
set_ctrl "$WBA_CTRL" 0
set_ctrl "$WB_CTRL" "$WB"
# 4) Kill backlight compensation — it fights us under a bright window.
set_ctrl "$BLC_CTRL" 0

echo "lock_camera: readback ---------------------------------------------------"
for c in "$AE_CTRL" "$EXP_CTRL" "$GAIN_CTRL" "$WBA_CTRL" "$WB_CTRL" "$BLC_CTRL"; do
    [ -n "$c" ] && v4l2-ctl -d "$DEV" --get-ctrl "$c" 2>/dev/null | sed 's/^/  /'
done
echo "lock_camera: done"
