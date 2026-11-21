"""A small persistent cache for NER and translation results, so analysing the same text or file again is instant.

Entries live in one SQLite file (default ~/.cache/annotation-tool/results.sqlite, outside any synced folder). The key is a
sha256 of everything that changes the answer (what was computed, the text, the language pair, the model and its settings),
so a changed setting never returns a stale result. The cache is only ever an optimisation: every error here is treated as
a cache miss and never reaches the caller.

    RESULT_CACHE_PATH      file to use; "off" disables the cache
    RESULT_CACHE_MAX_ROWS  entries kept (oldest are dropped), default 50000

The stored values contain pieces of the processed text (entities, translations). They stay on this machine; delete the
file to clear them.
"""
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2                 # bump when the stored format OR the logic that produced a result changes, so old entries are ignored
                                   # (2: Chinese/Japanese entities are no longer grown to the whole clause)
_lock = threading.Lock()
_warned = False


def _path() -> Optional[Path]:
    raw = os.getenv("RESULT_CACHE_PATH")
    if raw is not None and raw.strip().lower() in ("off", "none", "0", "false"):
        return None
    return Path(raw) if raw and raw.strip() else Path.home() / ".cache" / "annotation-tool" / "results.sqlite"


def _max_rows() -> int:
    try:
        return max(100, int(os.getenv("RESULT_CACHE_MAX_ROWS", "50000")))
    except ValueError:
        return 50000


def make_key(*parts) -> str:
    return hashlib.sha256(json.dumps([SCHEMA_VERSION, *parts], ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)")
    return conn


def _problem(e: Exception):
    global _warned
    if not _warned:
        _warned = True
        logger.warning(f"Result cache disabled for this error (results are simply recomputed): {e}")


def get(key: str) -> Optional[str]:
    path = _path()
    if path is None or not path.exists():
        return None
    try:
        with _lock:
            conn = _connect(path)
            try:
                row = conn.execute("SELECT value FROM results WHERE key = ?", (key,)).fetchone()
            finally:
                conn.close()
        return row[0] if row else None
    except Exception as e:                     # a broken cache must never break a request
        _problem(e)
        return None


def put(key: str, value: str) -> None:
    path = _path()
    if path is None:
        return
    try:
        with _lock:
            conn = _connect(path)
            try:
                conn.execute("INSERT OR REPLACE INTO results (key, value, created) VALUES (?, ?, ?)", (key, value, time.time()))
                limit = _max_rows()
                (count,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
                if count > limit:                       # drop the oldest tenth
                    conn.execute("DELETE FROM results WHERE key IN (SELECT key FROM results ORDER BY created LIMIT ?)",
                                 (max(1, count - limit + limit // 10),))
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        _problem(e)


def get_json(key: str):
    raw = get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def put_json(key: str, value) -> None:
    put(key, json.dumps(value, ensure_ascii=False))
