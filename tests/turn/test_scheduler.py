import asyncio
import logging

import pytest

from banbu.turn.scheduler import TurnScheduler, proactive_key, reactive_key


def test_thread_keys_are_stable() -> None:
    assert proactive_key("home_a", "scene_1") == "proactive:home_a:scene_1"
    assert reactive_key("home_a", "user_1") == "reactive:home_a:user_1"


@pytest.mark.asyncio
async def test_same_reactive_thread_runs_in_submission_order() -> None:
    scheduler = TurnScheduler()
    events: list[str] = []

    async def job(label: str) -> None:
        events.append(f"{label}:start")
        await asyncio.sleep(0.01)
        events.append(f"{label}:end")

    tasks = [
        scheduler.submit_reactive(home_id="home_a", user_id="user_1", job_factory=lambda: job("first")),
        scheduler.submit_reactive(home_id="home_a", user_id="user_1", job_factory=lambda: job("second")),
    ]

    await asyncio.gather(*tasks)

    assert events == ["first:start", "first:end", "second:start", "second:end"]


@pytest.mark.asyncio
async def test_different_proactive_threads_can_overlap() -> None:
    scheduler = TurnScheduler()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    events: list[str] = []

    async def first_job() -> None:
        events.append("first:start")
        first_started.set()
        await second_started.wait()
        events.append("first:end")

    async def second_job() -> None:
        await first_started.wait()
        events.append("second:start")
        second_started.set()
        events.append("second:end")

    tasks = [
        scheduler.submit_proactive(home_id="home_a", scene_id="scene_1", job_factory=first_job),
        scheduler.submit_proactive(home_id="home_a", scene_id="scene_2", job_factory=second_job),
    ]

    await asyncio.gather(*tasks)

    assert events == ["first:start", "second:start", "second:end", "first:end"]


@pytest.mark.asyncio
async def test_job_errors_are_logged_and_do_not_block_next_job(caplog) -> None:
    scheduler = TurnScheduler()
    events: list[str] = []

    async def bad_job() -> None:
        events.append("bad")
        raise RuntimeError("boom")

    async def good_job() -> None:
        events.append("good")

    with caplog.at_level(logging.ERROR, logger="banbu.turn.scheduler"):
        tasks = [
            scheduler.submit_reactive(home_id="home_a", user_id="user_1", job_factory=bad_job),
            scheduler.submit_reactive(home_id="home_a", user_id="user_1", job_factory=good_job),
        ]
        await asyncio.gather(*tasks)

    assert events == ["bad", "good"]
    assert "turn scheduler job failed key=reactive:home_a:user_1" in caplog.text
