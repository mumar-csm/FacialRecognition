#!/usr/bin/env python3
"""Register a new kiosk device with the central tier.

Generates a 32-byte URL-safe API key, stores its sha256 hash in `devices`, and
prints the plaintext key to stdout EXACTLY ONCE. The operator copies the
printed key into the Pi's /etc/fr-kiosk/secrets.env during SD imaging — central
never sees the plaintext again. If the key is lost, re-register the device
(manual rotation policy until ~100 stores; see project_central_tier_topology).

Usage (from inside central/, with conda env active and .env sourced via make):
    python -m tools.register_device --device-id store001-pi01 --store-id store001
"""

import argparse
import asyncio
import hashlib
import secrets
import sys

from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from app import db
from app.models import devices


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register a kiosk device with central.")
    p.add_argument(
        "--device-id",
        required=True,
        help="Unique device identifier, e.g. 'store001-pi01'. Must be globally unique.",
    )
    p.add_argument(
        "--store-id",
        required=True,
        help="Store this device belongs to, e.g. 'store001'. Multiple devices can share a store_id.",
    )
    return p.parse_args()


async def _insert_device(device_id: str, store_id: str, api_key_hash: str) -> None:
    async with db.engine().begin() as conn:
        await conn.execute(
            insert(devices).values(
                device_id=device_id,
                store_id=store_id,
                api_key_hash=api_key_hash,
            )
        )


async def _main() -> int:
    args = _parse_args()

    # Generate first so an IntegrityError on insert doesn't waste a key — though
    # since we never persist the plaintext, a "wasted" key is just entropy on the
    # floor. The point is to keep the secret generation close to the DB write.
    api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(api_key.encode("ascii")).hexdigest()

    try:
        await _insert_device(args.device_id, args.store_id, api_key_hash)
    except IntegrityError as e:
        print(
            f"error: device_id '{args.device_id}' already exists "
            f"(or api_key_hash collision, which would be astronomically unlikely).",
            file=sys.stderr,
        )
        print(f"  DB detail: {e.orig}", file=sys.stderr)
        return 1
    finally:
        await db.dispose()

    print()
    print("  Device registered.")
    print(f"    device_id : {args.device_id}")
    print(f"    store_id  : {args.store_id}")
    print(f"    api_key   : {api_key}")
    print()
    print("  Save the api_key NOW — central only stored the sha256 hash and will never")
    print("  show this value again. Paste it into the Pi's /etc/fr-kiosk/secrets.env as:")
    print("    CENTRAL_API_KEY=" + api_key)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
