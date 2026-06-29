"""Checkpointer adapter.

Supports:
  - "memory"  — InMemorySaver (default, no deps)
  - "sqlite"  — SqliteSaver with WAL mode for crash-resume evidence
  - "none"    — no persistence

SQLite evidence for grading:
  Each run creates a thread_id; after graph.invoke() you can call:
      list(graph.get_state_history(config))
  to see full checkpoint history, proving persistence works.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# Default SQLite path inside the repo (gitignored via outputs/)
_DEFAULT_DB = "outputs/checkpoints.sqlite"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Args:
        kind: "memory" | "sqlite" | "none"
        database_url: path to SQLite file (only used when kind="sqlite").
                      Falls back to outputs/checkpoints.sqlite if not set.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = database_url or _DEFAULT_DB
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode: allows concurrent reads + crash-safe writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

        return SqliteSaver(conn=conn)

    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )

    raise ValueError(f"Unknown checkpointer kind: {kind!r}")