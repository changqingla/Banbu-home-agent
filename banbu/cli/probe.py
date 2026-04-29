"""Phase-1 verification CLI.

Loads .env + devices.yaml, reconciles with the IoT platform, configures
emergency-keyword push (use --no-configure-emergency to skip), and prints:
  - reconciliation table (friendly_name → local_id, capabilities check)
  - latest snapshot per managed device

Exit codes:
  0 — all green
  1 — boot validation failed (missing devices, bad capabilities, etc.)
  2 — IoT platform unreachable
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from banbu.adapters.iot_client import IoTClient, IoTError
from banbu.config.settings import get_settings
from banbu.devices.registry import RegistryError, build_registry
from banbu.state.snapshot_cache import SnapshotCache


def _fmt_payload(payload: dict, max_chars: int = 200) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


async def _run(configure_emergency: bool) -> int:
    settings = get_settings()
    print(f"IoT base URL: {settings.iot_base_url}")
    print(f"devices file: {settings.devices_file}")
    print(f"registry strict: {settings.registry_strict}")
    print()

    async with IoTClient(settings) as client:
        try:
            resolver = await build_registry(
                client,
                settings.devices_file,
                configure_emergency=configure_emergency,
                strict=settings.registry_strict,
            )
        except RegistryError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1
        except IoTError as e:
            print(f"[FAIL] IoT platform error: {e}", file=sys.stderr)
            return 2

        print(f"{'friendly_name':<28} {'local_id':>8}  {'role':<16} caps")
        print("-" * 90)
        for dev in resolver.all():
            caps_preview = ", ".join(sorted(dev.capabilities)[:6])
            if len(dev.capabilities) > 6:
                caps_preview += ", …"
            print(
                f"{dev.spec.friendly_name:<28} {dev.local_id:>8}  {dev.spec.role:<16} {caps_preview}"
            )
        print()
        if resolver.skipped_missing_devices:
            print("Skipped missing devices (declared in devices.yaml but absent from IoT listing):")
            for name in resolver.skipped_missing_devices:
                print(f"  - {name}")
            print()

        cache = SnapshotCache(resolver)
        await cache.bootstrap(client)
        print("Latest snapshots (from /devices/allinfo):")
        print("-" * 90)
        for dev in resolver.all():
            snap = cache.get(dev.local_id)
            if snap is None:
                print(f"  {dev.spec.friendly_name}: <no payload yet>")
            else:
                print(f"  {dev.spec.friendly_name}: {_fmt_payload(snap.payload)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Banbu phase-1 device probe")
    parser.add_argument(
        "--no-configure-emergency",
        action="store_true",
        help="Skip writing report-config (use for read-only diagnosis)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    exit_code = asyncio.run(_run(configure_emergency=not args.no_configure_emergency))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
