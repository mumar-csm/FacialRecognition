#!/usr/bin/env python3
"""Provision a kiosk Pi: write its device config + env file.

Run at HQ during SD imaging (or the first provisioning of a freshly-cloned card).
Writes the two files that `kiosk_server.py:_apply_device_config` and the
`fr-kiosk.service` systemd unit expect:

  <output-dir>/device.json   (mode 644, non-secret)  — device_id, store_id,
                              central_url + any interval overrides. Read by the
                              kiosk via --device-config.
  <output-dir>/secrets.env   (mode 600, secret)      — CENTRAL_API_KEY (+ optional
                              ENROLLMENT_PIN). Read by systemd via EnvironmentFile=.

The API key is the plaintext printed ONCE by `central/tools/register_device.py`.
It is never accepted as a plain CLI flag (would land in shell history / `ps`).
Supply it via the FR_PROVISION_API_KEY env var, or you'll be prompted for it.

Usage (default writes to /etc/fr-kiosk, needs sudo):
    FR_PROVISION_API_KEY=<key> sudo -E python tools/provision_device.py \
        --device-id store001-pi01 --store-id store001 \
        --central-url https://fr-central.example.ts.net

Dry-run / image-build staging (no root needed):
    FR_PROVISION_API_KEY=<key> python tools/provision_device.py \
        --device-id store001-pi01 --store-id store001 \
        --central-url https://fr-central.example.ts.net \
        --output-dir /tmp/fr-kiosk-stage
"""

import argparse
import getpass
import json
import os
import sys

DEFAULT_OUTPUT_DIR = "/etc/fr-kiosk"

# Keys allowed in device.json. Mirrors the fields _apply_device_config reads.
# Only the three identity fields are required; intervals/batch are written only
# when the operator overrides them, so the kiosk's own defaults stay in charge.
_DEVICE_JSON_OPTIONAL = ("sync_interval_seconds", "sync_batch_size", "roster_interval_seconds")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Provision a kiosk Pi's device.json + secrets.env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device-id", required=True,
                   help="Stable identifier for this Pi, e.g. 'store001-pi01'. "
                        "Must match the device_id registered with central.")
    p.add_argument("--store-id", required=True,
                   help="Store this device belongs to, e.g. 'store001'.")
    p.add_argument("--central-url", required=True,
                   help="Base URL of the central tier, e.g. "
                        "'https://fr-central.example.ts.net' (tailnet).")
    p.add_argument("--sync-interval-seconds", type=int, default=None,
                   help="Override outbox drain interval (kiosk default 1800).")
    p.add_argument("--sync-batch-size", type=int, default=None,
                   help="Override outbox upload batch size (kiosk default 50).")
    p.add_argument("--roster-interval-seconds", type=int, default=None,
                   help="Override roster poll interval (kiosk default 1800).")
    p.add_argument("--enrollment-pin", default=None,
                   help="Manager PIN for enroll/delete. Written to secrets.env "
                        "(mode 600) as ENROLLMENT_PIN so it stays out of ExecStart/ps.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Directory to write into (default {DEFAULT_OUTPUT_DIR}). "
                        "Use a staging path for image builds / dry-runs.")
    return p.parse_args()


def _resolve_api_key() -> str:
    """The plaintext CENTRAL_API_KEY — from env, else an interactive prompt.

    Never a CLI flag: keeps the secret out of shell history and `ps aux`.
    """
    key = os.environ.get("FR_PROVISION_API_KEY")
    if key:
        return key.strip()
    key = getpass.getpass("CENTRAL_API_KEY (paste the key register_device printed): ").strip()
    if not key:
        print("error: no API key provided (env FR_PROVISION_API_KEY or prompt).",
              file=sys.stderr)
        sys.exit(1)
    return key


def _build_device_config(args: argparse.Namespace) -> dict:
    cfg = {
        "device_id": args.device_id,
        "store_id": args.store_id,
        "central_url": args.central_url,
    }
    for field in _DEVICE_JSON_OPTIONAL:
        val = getattr(args, field)
        if val is not None:
            cfg[field] = val
    return cfg


def _write_file(path: str, content: str, mode: int) -> None:
    """Write `content` to `path` and force `mode` (umask-independent).

    Opens with the target mode so the secret never briefly exists world-readable,
    then chmod to be certain even if the file already existed.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, mode)


def main() -> int:
    args = _parse_args()
    api_key = _resolve_api_key()

    out_dir = args.output_dir
    try:
        os.makedirs(out_dir, mode=0o755, exist_ok=True)
    except OSError as e:
        print(f"error: cannot create {out_dir}: {e}\n"
              f"  (writing to {DEFAULT_OUTPUT_DIR} needs sudo; or pass --output-dir)",
              file=sys.stderr)
        return 1

    device_json_path = os.path.join(out_dir, "device.json")
    secrets_env_path = os.path.join(out_dir, "secrets.env")

    cfg = _build_device_config(args)
    _write_file(device_json_path, json.dumps(cfg, indent=2) + "\n", 0o644)

    lines = [
        "# Secrets for fr-kiosk.service (systemd EnvironmentFile). Mode 600.",
        "# Generated by tools/provision_device.py — do NOT commit a real copy.",
        f"CENTRAL_API_KEY={api_key}",
    ]
    if args.enrollment_pin is not None:
        lines.append(f"ENROLLMENT_PIN={args.enrollment_pin}")
    _write_file(secrets_env_path, "\n".join(lines) + "\n", 0o600)

    print()
    print("  Device provisioned.")
    print(f"    {device_json_path}  (mode 644)")
    for k, v in cfg.items():
        print(f"        {k}: {v}")
    print(f"    {secrets_env_path}  (mode 600)")
    print(f"        CENTRAL_API_KEY: <{len(api_key)} chars, not echoed>")
    if args.enrollment_pin is not None:
        print("        ENROLLMENT_PIN: <set, not echoed>")
    print()
    print("  Next: sudo systemctl enable --now fr-kiosk.service")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
