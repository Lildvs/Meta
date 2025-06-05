from __future__ import annotations

"""Lightweight async embedding / snippet cache backed by SQLite.

Design goals
------------
* single-file DB; safe for low-concurrency async usage
* simple *get_or_fetch* API – callers supply the key and a coroutine to
  compute the value if missing
* stores JSON serialisation of arbitrary python objects plus a placeholder
  BLOB column for future vector embeddings
* keeps global hit / miss counters so orchestrator can log cache ratio
"""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_DB_PATH = Path(__file__).with_suffix(".sqlite3")


class EmbeddingCache:  # noqa: D401 – simple wrapper
    """Simple SQLite-based cache for arbitrary JSON-serialisable objects."""

    _instance: "EmbeddingCache | None" = None

    def __new__(cls):  # singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()  # type: ignore[attr-defined]
        return cls._instance

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------

    async def get_or_fetch(self, key: str, fetcher: Callable[[], Awaitable[T]]) -> T:  # noqa: D401
        """Return cached value for *key* or fetch and save it.

        Parameters
        ----------
        key:
            Unique deterministic string (e.g. f"{depth}:{query}")
        fetcher:
            Async function that produces the value if *key* missing
        """
        # First: try cache synchronously to keep event-loop free
        hit, serialized = await _run_db(self._get_value, key)
        if hit:
            EmbeddingCache._hits += 1
            return json.loads(serialized)  # type: ignore[return-value]

        # Miss → compute
        EmbeddingCache._misses += 1
        value = await fetcher()
        try:
            await _run_db(self._set_value, key, json.dumps(value))
        except Exception as err:  # noqa: BLE001
            logger.warning("Cache write failed: %s", err)
        return value

    # -----------------------------------------------------
    # stats helpers
    # -----------------------------------------------------

    @classmethod
    def hit_ratio(cls) -> float:
        total = cls._hits + cls._misses
        return 0.0 if total == 0 else cls._hits / total

    # -----------------------------------------------------
    # internal DB helpers (executed in thread)
    # -----------------------------------------------------

    _hits: int = 0
    _misses: int = 0

    def _init_db(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_DB_PATH) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    vector BLOB,
                    json  TEXT
                )
                """
            )
            con.commit()

    # --- db ops
    @staticmethod
    def _get_value(con: sqlite3.Connection, key: str) -> tuple[bool, str]:
        cur = con.execute("SELECT json FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return False, ""
        return True, row[0]  # type: ignore[index]

    @staticmethod
    def _set_value(con: sqlite3.Connection, key: str, json_val: str) -> None:
        con.execute(
            "INSERT OR REPLACE INTO cache (key, json) VALUES (?, ?)", (key, json_val)
        )
        con.commit()


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

async def _run_db(fn: Callable[..., T], *args) -> T:  # noqa: D401
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _db_worker, fn, *args)


def _db_worker(fn: Callable[..., T], *args) -> T:
    with sqlite3.connect(_DB_PATH) as con:
        return fn(con, *args)