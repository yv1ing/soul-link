import math
import time
import asyncio
import sqlite3
import threading
from config import settings

_DAY_SECONDS = 86400.0

_NEVER_FORGET_CATEGORIES = frozenset({"profile"})

_CATEGORY_BASE_IMPORTANCE: dict[str, float] = {
    "profile":     1.0,
    "preferences": 0.7,
    "entities":    0.6,
    "patterns":    0.5,
    "events":      0.4,
    "cases":       0.4,
}


class MemoryDecay:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        self._base_stability = settings.memory_decay_base_stability
        self._w_imp = settings.memory_decay_importance_weight
        self._w_acc = settings.memory_decay_access_weight
        self._soft_threshold = settings.memory_decay_soft_threshold
        self._hard_threshold = settings.memory_decay_hard_threshold

        self._conn = conn
        self._lock = lock

        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_meta (
                    uri TEXT PRIMARY KEY,
                    category TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed REAL NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            self._conn.commit()

    def _compute_stability(self, importance: float, access_count: int) -> float:
        return self._base_stability * (1.0 + importance * self._w_imp + math.log1p(access_count) * self._w_acc)

    def _compute_retention(self, elapsed_days: float, importance: float, access_count: int) -> float:
        if elapsed_days <= 0:
            return 1.0

        stability = self._compute_stability(importance, access_count)
        return math.exp(-elapsed_days / stability)

    async def record_accesses(self, memories: list[dict]) -> None:
        now = time.time()

        def _do():
            with self._lock:
                for m in memories:
                    uri = m.get("uri", "")
                    if not uri:
                        continue

                    category = m.get("category", "")
                    base_imp = _CATEGORY_BASE_IMPORTANCE.get(category, 0.5)

                    self._conn.execute(
                        """
                            INSERT INTO memory_meta (uri, category, importance, access_count, last_accessed, created_at)
                            VALUES (?, ?, ?, 1, ?, ?) ON CONFLICT(uri) DO UPDATE SET access_count = access_count + 1, last_accessed = excluded.last_accessed
                        """,
                        (uri, category, base_imp, now, now),
                    )

                self._conn.commit()

        await asyncio.to_thread(_do)

    async def apply_decay(self, memories: list[dict]) -> list[dict]:
        if not memories:
            return []

        now = time.time()

        def _do():
            with self._lock:
                results = []

                for m in memories:
                    uri = m.get("uri", "")
                    category = m.get("category", "")

                    if category in _NEVER_FORGET_CATEGORIES:
                        results.append(m)
                        continue

                    row = self._conn.execute(
                        "SELECT importance, access_count, last_accessed FROM memory_meta WHERE uri=?",
                        (uri,),
                    ).fetchone()

                    if row:
                        importance, access_count, last_accessed = row
                        elapsed_days = (now - last_accessed) / _DAY_SECONDS
                    else:
                        importance = _CATEGORY_BASE_IMPORTANCE.get(category, 0.5)
                        access_count = 0
                        elapsed_days = 0.0

                        self._conn.execute(
                            "INSERT OR IGNORE INTO memory_meta (uri, category, importance, access_count, last_accessed, created_at) VALUES (?, ?, ?, 0, ?, ?)",
                            (uri, category, importance, now, now),
                        )

                    retention = self._compute_retention(elapsed_days, importance, access_count)

                    if retention < self._soft_threshold:
                        continue

                    adjusted = m.copy()
                    adjusted["retention"] = retention
                    results.append(adjusted)

                self._conn.commit()

            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results

        return await asyncio.to_thread(_do)

    async def collect_forgotten(self) -> list[dict]:
        if self._hard_threshold <= 0:
            return []

        now = time.time()

        def _do():
            with self._lock:
                placeholders = ",".join("?" * len(_NEVER_FORGET_CATEGORIES))
                rows = self._conn.execute(
                    f"SELECT uri, category, importance, access_count, last_accessed FROM memory_meta WHERE category NOT IN ({placeholders})",
                    tuple(_NEVER_FORGET_CATEGORIES),
                ).fetchall()

                forgotten = []
                for uri, category, importance, access_count, last_accessed in rows:
                    elapsed_days = (now - last_accessed) / _DAY_SECONDS
                    retention = self._compute_retention(elapsed_days, importance, access_count)

                    if retention < self._hard_threshold:
                        forgotten.append({
                            "uri": uri,
                            "category": category,
                            "retention": retention,
                        })

            return forgotten

        return await asyncio.to_thread(_do)

    async def purge(self, uris: list[str]) -> None:
        if not uris:
            return

        def _do():
            with self._lock:
                placeholders = ",".join("?" * len(uris))
                self._conn.execute(f"DELETE FROM memory_meta WHERE uri IN ({placeholders})", uris)
                self._conn.commit()

        await asyncio.to_thread(_do)
