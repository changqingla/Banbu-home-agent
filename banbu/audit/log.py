"""Append-only SQLite audit log.

One row per significant event in a turn:
  trigger_id : groups all rows belonging to the same Turn (proactive or reactive)
  scene_id   : nullable for reactive turns
  kind       : 'trigger' | 'agent_request' | 'agent_response' | 'tool_call' |
               'tool_result' | 'execute' | 'execute_result' | 'cooldown_set'
  payload    : JSON-encoded body
  created_at : unix seconds
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5.0, isolation_level=None)

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS audit (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       trigger_id TEXT,
                       scene_id TEXT,
                       kind TEXT NOT NULL,
                       payload TEXT NOT NULL,
                       created_at REAL NOT NULL
                   )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_trigger ON audit (trigger_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_kind ON audit (kind)")

    def write(
        self,
        kind: str,
        payload: Any,
        *,
        trigger_id: str | None = None,
        scene_id: str | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO audit (trigger_id, scene_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
                (trigger_id, scene_id, kind, body, time.time()),
            )

    def by_trigger(self, trigger_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, trigger_id, scene_id, kind, payload, created_at FROM audit WHERE trigger_id=? ORDER BY id",
                (trigger_id,),
            )
            return [
                {
                    "id": r[0],
                    "trigger_id": r[1],
                    "scene_id": r[2],
                    "kind": r[3],
                    "payload": json.loads(r[4]),
                    "created_at": r[5],
                }
                for r in cur.fetchall()
            ]
