import pytest

from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.scenes.definition import Precondition
from banbu.scenes.runtime.conditions import PreconditionFailed, check_precondition, eval_op
from banbu.state.snapshot_cache import SnapshotCache


def _cache() -> SnapshotCache:
    resolver = DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(friendly_name="entry_light", role="light_switch"),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state", "illuminance"},
            )
        ]
    )
    cache = SnapshotCache(resolver)
    cache.update(2, {"state": "OFF", "illuminance": 12}, source="test")
    return cache


def test_eval_op_supported_operations() -> None:
    assert eval_op("eq", 1, 1) is True
    assert eval_op("neq", 1, 2) is True
    assert eval_op("lt", 1, 2) is True
    assert eval_op("lte", 2, 2) is True
    assert eval_op("gt", 3, 2) is True
    assert eval_op("gte", 3, 3) is True
    assert eval_op("in", "a", ["a", "b"]) is True


def test_check_precondition_reads_snapshot_field() -> None:
    ok, summary = check_precondition(
        Precondition(device="entry_light", field="payload.illuminance", op="lt", value=30),
        _cache(),
    )

    assert ok is True
    assert "entry_light.payload.illuminance=12 lt 30 -> True" == summary


def test_check_precondition_missing_fail_raises() -> None:
    with pytest.raises(PreconditionFailed):
        check_precondition(
            Precondition(
                device="entry_light",
                field="payload.missing",
                op="eq",
                value=True,
                on_missing="fail",
            ),
            _cache(),
        )
