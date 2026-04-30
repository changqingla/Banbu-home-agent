"""Reactive CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from banbu.adapters.iot_client import IoTClient, IoTError
from banbu.audit.log import AuditLog
from banbu.config.settings import get_settings
from banbu.control.executor import Executor
from banbu.control.plane import ControlPlane
from banbu.devices.registry import RegistryError, build_registry
from banbu.policy import PolicyLoadError, load_policy
from banbu.reactive.agent_runner import ReactiveAgentRunner, result_payload
from banbu.scenes.loader import SceneLoadError, load_scenes
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.builder import from_reactive


async def _run_utterance(utterance: str, *, user_id: str) -> int:
    settings = get_settings()

    async with IoTClient(settings) as client:
        try:
            resolver = await build_registry(
                client,
                settings.devices_file,
                configure_emergency=False,
            )
        except RegistryError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1
        except IoTError as e:
            print(f"[FAIL] IoT platform error: {e}", file=sys.stderr)
            return 2

        try:
            scenes = load_scenes(settings.scenes_dir, resolver)
        except SceneLoadError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1

        try:
            policy = load_policy(settings.policy_file)
        except PolicyLoadError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1

        audit = AuditLog(settings.db_path)
        cache = SnapshotCache(resolver)
        await cache.bootstrap(client)
        control = ControlPlane(Executor(client), resolver, audit, policy, cache=cache)
        runner = ReactiveAgentRunner(
            settings=settings,
            resolver=resolver,
            control=control,
            audit=audit,
            cache=cache,
            scenes=scenes,
        )
        turn = from_reactive(
            utterance,
            home_id=settings.home_id,
            user_id=user_id,
            source="cli",
        )
        result = await runner.run(turn)
        print(json.dumps(result_payload(result), ensure_ascii=False, sort_keys=True))
        return 0 if result.ok else 1


def _read_stdin_utterance() -> str:
    if sys.stdin.isatty():
        return input("> ").strip()
    return sys.stdin.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Banbu reactive user-turn CLI")
    parser.add_argument("utterance", nargs="*", help="User request, e.g. 打开玄关灯")
    parser.add_argument("--user-id", default="cli_user", help="Reactive user id for conversation threading")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    utterance = " ".join(args.utterance).strip() or _read_stdin_utterance()
    if not utterance:
        print("[FAIL] reactive utterance must not be empty", file=sys.stderr)
        sys.exit(1)

    sys.exit(asyncio.run(_run_utterance(utterance, user_id=args.user_id)))


if __name__ == "__main__":
    main()
