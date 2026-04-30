"""Banbu FastAPI app entry point.

Boot sequence:
  1. Load .env via pydantic-settings.
  2. Build device registry (validates devices.yaml against the IoT platform).
  3. Bootstrap snapshot cache from /devices/allinfo.
  4. Load scenes + reverse index.
  5. Wire control plane + agent loop.
  6. Mount webhook route, start fallback poller.
  7. Print the public webhook URL so it can be configured on the IoT side.

Run:
    .venv/bin/python -m banbu.main
"""
from __future__ import annotations

import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI

from banbu.adapters.iot_client import IoTClient
from banbu.agent.loop import AgentLoop, AgentResult
from banbu.audit.log import AuditLog
from banbu.config.settings import Settings, get_settings
from banbu.context.assembler import assemble_blocks
from banbu.context.pilot import optimize as pilot_optimize
from banbu.context.selector import select as select_context
from banbu.control.executor import Executor
from banbu.control.plane import ControlPlane
from banbu.devices.registry import build_registry
from banbu.devices.resolver import DeviceResolver
from banbu.dispatcher import Dispatcher
from banbu.ingest.poller import FallbackPoller
from banbu.im.router import make_feishu_sdk_service, make_router as make_im_router
from banbu.ingest.webhook import make_router as make_ingest_router
from banbu.policy import load_policy
from banbu.reactive.runner import ReactiveRunner
from banbu.scenes.definition import Scene
from banbu.scenes.loader import load_scenes
from banbu.scenes.reverse_index import build_reverse_index
from banbu.state.feedback import FeedbackEntry, FeedbackOutcome, FeedbackStore
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.builder import from_trigger
from banbu.turn.model import ProactiveTrigger
from banbu.turn.scheduler import TurnScheduler
from banbu.vision.service import VisionService

log = logging.getLogger(__name__)


def _feedback_outcome(agent_result: AgentResult, tool_results: list[dict]) -> tuple[FeedbackOutcome, str]:
    if agent_result.error:
        return "agent_error", agent_result.error
    if any(not r.get("ok") for r in tool_results):
        return "failure", "one or more execute_plan calls failed"
    if agent_result.executed:
        return "success", f"executed {len(agent_result.executed)} action(s)"
    return "skipped", "agent returned no executable actions"


def _lan_ip(target: str = "192.168.1.78") -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _make_handle_trigger(
    *,
    settings: Settings,
    resolver: DeviceResolver,
    cache: SnapshotCache,
    scenes_by_id: dict[str, Scene],
    agent: AgentLoop,
    control: ControlPlane,
    dispatcher: Dispatcher,
    audit: AuditLog,
    feedback_store: FeedbackStore,
):
    async def handle(trigger: ProactiveTrigger) -> None:
        scene = scenes_by_id.get(trigger.scene_id)
        if scene is None:
            log.warning("handle_trigger: unknown scene_id %s", trigger.scene_id)
            feedback_store.add(FeedbackEntry(
                home_id=trigger.home_id,
                scene_id=trigger.scene_id,
                trigger_id=trigger.trigger_id,
                outcome="failure",
                summary="unknown scene_id",
            ))
            dispatcher.mark_executed(trigger.scene_id, success=False)
            return

        audit.write("trigger", trigger.__dict__,
                    trigger_id=trigger.trigger_id, scene_id=trigger.scene_id)

        turn = from_trigger(trigger)
        ctx = select_context(turn, scene, resolver, cache, feedback_store=feedback_store)
        blocks = assemble_blocks(ctx)
        messages = pilot_optimize(blocks, conversation_id=turn.conversation_id)
        tool_results: list[dict] = []

        async def on_execute(local_id: int, action: str, params: dict | None) -> dict:
            result = await control.execute(
                local_id, action, params,
                trigger_id=trigger.trigger_id,
                scene_id=trigger.scene_id,
                actor="proactive",
                home_id=trigger.home_id,
            )
            payload = {
                "ok": result.ok,
                "local_id": result.local_id,
                "action": result.action,
                "payload": result.payload,
                "error": result.error,
                "deduped": result.deduped,
            }
            tool_results.append(payload)
            return payload

        try:
            agent_result = await agent.run(
                messages,
                on_execute=on_execute,
                trigger_id=trigger.trigger_id,
                scene_id=trigger.scene_id,
            )
        except Exception:
            log.exception("agent run crashed for trigger %s", trigger.trigger_id)
            feedback_store.add(FeedbackEntry(
                home_id=trigger.home_id,
                scene_id=trigger.scene_id,
                trigger_id=trigger.trigger_id,
                outcome="agent_error",
                summary="agent run crashed",
            ))
            dispatcher.mark_executed(trigger.scene_id, success=False)
            return

        log.info(
            "agent done trigger=%s iters=%d executed=%d final=%r error=%s",
            trigger.trigger_id, agent_result.iterations,
            len(agent_result.executed), agent_result.final_message, agent_result.error,
        )

        success = bool(agent_result.executed) and not agent_result.error
        outcome, summary = _feedback_outcome(agent_result, tool_results)

        feedback_store.add(FeedbackEntry(
            home_id=trigger.home_id,
            scene_id=trigger.scene_id,
            trigger_id=trigger.trigger_id,
            outcome=outcome,
            summary=summary,
            details={
                "iterations": agent_result.iterations,
                "executed": agent_result.executed,
                "tool_results": tool_results,
                "final_message": agent_result.final_message,
                "error": agent_result.error,
            },
        ))
        dispatcher.mark_executed(
            trigger.scene_id,
            success=success,
            cooldown_seconds=scene.policy.cooldown_seconds if success else None,
        )

    return handle


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = IoTClient(settings)
    resolver = await build_registry(client, settings.devices_file, strict=settings.registry_strict)
    cache = SnapshotCache(resolver)
    await cache.bootstrap(client)

    scenes = load_scenes(settings.scenes_dir, resolver)
    scenes_by_id = {s.scene_id: s for s in scenes}
    reverse_index = build_reverse_index(scenes, resolver)
    log.info("reverse index built: %d (device,field) entries", len(reverse_index))

    audit = AuditLog(settings.db_path)
    executor = Executor(client)
    policy = load_policy(settings.policy_file)
    control = ControlPlane(
        executor,
        resolver,
        audit,
        policy,
        cache=cache,
        scene_priorities={scene.scene_id: scene.policy.priority for scene in scenes},
    )
    agent = AgentLoop(settings, audit)
    feedback_store = FeedbackStore()
    turn_scheduler = TurnScheduler()

    pending_handler = {"fn": None}

    def _on_hit(trigger: ProactiveTrigger) -> None:
        handler = pending_handler["fn"]
        if handler is None:
            log.error("on_hit fired before handler was wired (this is a bug)")
            return
        log.info(
            "ProactiveTrigger emitted scene=%s id=%s home=%s",
            trigger.scene_id, trigger.trigger_id, trigger.home_id,
        )
        turn_scheduler.submit_proactive(
            home_id=trigger.home_id,
            scene_id=trigger.scene_id,
            job_factory=lambda trigger=trigger: handler(trigger),
        )

    dispatcher = Dispatcher(scenes, reverse_index, cache, home_id=settings.home_id, on_hit=_on_hit)

    pending_handler["fn"] = _make_handle_trigger(
        settings=settings,
        resolver=resolver,
        cache=cache,
        scenes_by_id=scenes_by_id,
        agent=agent,
        control=control,
        dispatcher=dispatcher,
        audit=audit,
        feedback_store=feedback_store,
    )

    app.include_router(make_ingest_router(
        path=settings.webhook_path,
        resolver=resolver,
        cache=cache,
        on_event=dispatcher.on_event,
    ))

    reactive_runner = ReactiveRunner(
        resolver=resolver,
        control=control,
        audit=audit,
        scenes=scenes,
    )
    app.include_router(make_im_router(
        settings=settings,
        runner=reactive_runner,
        scheduler=turn_scheduler,
    ))
    feishu_sdk_service = make_feishu_sdk_service(
        settings=settings,
        runner=reactive_runner,
        scheduler=turn_scheduler,
    )
    feishu_sdk_service.start()

    poller = FallbackPoller(
        client, resolver, cache,
        interval_seconds=settings.fallback_poll_seconds,
        tick_timeout_seconds=settings.iot_timeout_seconds + 2,
        on_event=dispatcher.on_event,
        on_tick=dispatcher.on_tick,
    )
    poller.start()
    vision_service = VisionService(settings, scenes)
    vision_service.start()

    public_url = f"http://{_lan_ip()}:{settings.port}{settings.webhook_path}"
    log.info("=" * 70)
    log.info("Banbu webhook ready. Configure the IoT platform to POST to:")
    log.info("  %s", public_url)
    log.info("=" * 70)

    app.state.iot_client = client
    app.state.resolver = resolver
    app.state.cache = cache
    app.state.poller = poller
    app.state.vision_service = vision_service
    app.state.dispatcher = dispatcher
    app.state.scenes = scenes
    app.state.audit = audit
    app.state.control = control
    app.state.agent = agent
    app.state.reactive_runner = reactive_runner
    app.state.feishu_sdk_service = feishu_sdk_service
    app.state.feedback_store = feedback_store
    app.state.turn_scheduler = turn_scheduler

    try:
        yield
    finally:
        await feishu_sdk_service.stop()
        await turn_scheduler.aclose()
        await vision_service.stop()
        await poller.stop()
        await client.aclose()


app = FastAPI(title="Banbu", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/snapshots")
def snapshots() -> dict:
    cache: SnapshotCache = app.state.cache
    return {
        "snapshots": [
            {
                "friendly_name": s.friendly_name,
                "local_id": s.local_id,
                "payload": s.payload,
                "updated_at": s.updated_at,
                "source": s.source,
            }
            for s in cache.all()
        ]
    }


@app.get("/audit/{trigger_id}")
def audit_for(trigger_id: str) -> dict:
    audit: AuditLog = app.state.audit
    return {"trigger_id": trigger_id, "rows": audit.by_trigger(trigger_id)}


def main() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "banbu.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
