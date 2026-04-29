from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

JobFactory = Callable[[], Awaitable[None]]


def proactive_key(home_id: str, scene_id: str) -> str:
    return f"proactive:{home_id}:{scene_id}"


def reactive_key(home_id: str, user_id: str) -> str:
    return f"reactive:{home_id}:{user_id}"


class TurnScheduler:
    """Serialize turn handling per logical thread key."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, key: str, job_factory: JobFactory) -> asyncio.Task[None]:
        task = asyncio.create_task(self._run(key, job_factory), name=f"banbu-turn:{key}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def submit_proactive(
        self,
        *,
        home_id: str,
        scene_id: str,
        job_factory: JobFactory,
    ) -> asyncio.Task[None]:
        return self.submit(proactive_key(home_id, scene_id), job_factory)

    def submit_reactive(
        self,
        *,
        home_id: str,
        user_id: str,
        job_factory: JobFactory,
    ) -> asyncio.Task[None]:
        return self.submit(reactive_key(home_id, user_id), job_factory)

    async def run_serialized(self, key: str, job_factory: JobFactory) -> None:
        await self._run(key, job_factory)

    async def _run(self, key: str, job_factory: JobFactory) -> None:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock

        async with lock:
            try:
                await job_factory()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("turn scheduler job failed key=%s", key)

    async def aclose(self) -> None:
        tasks = list(self._tasks)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
