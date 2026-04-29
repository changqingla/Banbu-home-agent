from __future__ import annotations

from typing import Any

from banbu.scenes.definition import Precondition
from banbu.state.snapshot_cache import MISSING, SnapshotCache


class PreconditionFailed(RuntimeError):
    pass


def eval_op(op: str, lhs: Any, rhs: Any) -> bool:
    if op == "eq":
        return lhs == rhs
    if op == "neq":
        return lhs != rhs
    if op == "lt":
        return lhs < rhs
    if op == "lte":
        return lhs <= rhs
    if op == "gt":
        return lhs > rhs
    if op == "gte":
        return lhs >= rhs
    if op == "in":
        return lhs in rhs
    raise ValueError(f"unknown op: {op}")


def check_precondition(pre: Precondition, cache: SnapshotCache) -> tuple[bool, str]:
    value, _ts = cache.field(pre.device, pre.field)
    if value is MISSING:
        if pre.on_missing == "skip":
            return False, f"{pre.device}.{pre.field} missing -> skip"
        if pre.on_missing == "pass":
            return True, f"{pre.device}.{pre.field} missing -> pass"
        raise PreconditionFailed(f"{pre.device}.{pre.field} missing -> fail")
    try:
        ok = eval_op(pre.op, value, pre.value)
    except TypeError as e:
        return False, f"{pre.device}.{pre.field}={value!r} {pre.op} {pre.value!r} -> type error ({e})"
    return ok, f"{pre.device}.{pre.field}={value!r} {pre.op} {pre.value!r} -> {ok}"
