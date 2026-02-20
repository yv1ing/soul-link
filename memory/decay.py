import math
import time
import sqlite3
import threading
from config import settings


_DAY_SECONDS = 86400.0
_NEVER_FORGET = frozenset({"profile"})

CATEGORY_BASE_IMPORTANCE: dict[str, float] = {
    "profile":     1.0,
    "preferences": 0.7,
    "entities":    0.6,
    "patterns":    0.5,
    "events":      0.4,
    "cases":       0.4,
}


class MemoryDecay:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        self._base = settings.memory_decay_base_stability
        self._w_imp = settings.memory_decay_importance_weight
        self._w_acc = settings.memory_decay_access_weight
        self._soft = settings.memory_decay_soft_threshold
        self._hard = settings.memory_decay_hard_threshold
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
                created_at REAL NOT NULL)
            """)
            self._conn.commit()

    def _stability(self, importance: float, access_count: int) -> float:
        return self._base * (1.0 + importance * self._w_imp + math.log1p(access_count) * self._w_acc)

    def _retention(self, elapsed_days: float, importance: float, access_count: int) -> float:
        if elapsed_days <= 0:
            return 1.0

        return math.exp(-elapsed_days / self._stability(importance, access_count))

    def record_accesses(self, memories: list[dict]) -> None:
        now = time.time()
        params = []
        for m in memories:
            uri = m.get("uri", "")
            if not uri:
                continue
            cat = m.get("category", "")
            imp = CATEGORY_BASE_IMPORTANCE.get(cat, 0.5)
            params.append((uri, cat, imp, now, now))

        if not params:
            return

        with self._lock:
            self._conn.executemany(
                """
                    INSERT INTO memory_meta (uri, category, importance, access_count, last_accessed, created_at)
                    VALUES (?,?,?,1,?,?) ON CONFLICT(uri) DO UPDATE SET access_count = access_count + 1, last_accessed = excluded.last_accessed
                """,
                params
            )
            self._conn.commit()

    def apply_decay(self, memories: list[dict]) -> list[dict]:
        if not memories:
            return []

        now = time.time()
        results = []

        uris = [m.get("uri", "") for m in memories if m.get("uri")]

        with self._lock:
            # Batch fetch all metadata in one query
            meta_map: dict[str, tuple[float, int, float]] = {}
            if uris:
                ph = ",".join("?" * len(uris))
                rows = self._conn.execute(
                    f"SELECT uri, importance, access_count, last_accessed FROM memory_meta WHERE uri IN ({ph})", uris
                ).fetchall()
                meta_map = {r[0]: (r[1], r[2], r[3]) for r in rows}

            new_entries = []
            for m in memories:
                uri = m.get("uri", "")
                category = m.get("category", "")

                if category in _NEVER_FORGET:
                    results.append(m)
                    continue

                if uri in meta_map:
                    importance, access_count, last_accessed = meta_map[uri]
                    elapsed = (now - last_accessed) / _DAY_SECONDS
                else:
                    importance = CATEGORY_BASE_IMPORTANCE.get(category, 0.5)
                    access_count = 0
                    elapsed = 0.0
                    new_entries.append((uri, category, importance, now, now))

                retention = self._retention(elapsed, importance, access_count)
                if retention < self._soft:
                    continue

                adjusted = m.copy()
                adjusted["retention"] = retention
                results.append(adjusted)

            if new_entries:
                self._conn.executemany(
                    """INSERT OR IGNORE INTO memory_meta (uri, category, importance, access_count, last_accessed, created_at) VALUES (?,?,?,0,?,?)""", new_entries
                )
            self._conn.commit()

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    def collect_forgotten(self) -> list[dict]:
        if self._hard <= 0:
            return []

        now = time.time()
        with self._lock:
            ph = ",".join("?" * len(_NEVER_FORGET))
            rows = self._conn.execute(
                f"""
                    SELECT uri, category, importance, access_count, last_accessed FROM memory_meta WHERE category NOT IN ({ph})
                """,
                tuple(_NEVER_FORGET)
            ).fetchall()

        forgotten = []
        for uri, cat, imp, acc, la in rows:
            ret = self._retention((now - la) / _DAY_SECONDS, imp, acc)
            if ret < self._hard:
                forgotten.append({"uri": uri, "category": cat, "retention": ret})

        return forgotten

    def get_fading(self, retention_max: float = 0.5, limit: int = 10) -> list[dict]:
        now = time.time()
        with self._lock:
            ph = ",".join("?" * len(_NEVER_FORGET))
            rows = self._conn.execute(
                f"SELECT uri, category, importance, access_count, last_accessed FROM memory_meta WHERE category NOT IN ({ph})",
                tuple(_NEVER_FORGET)
            ).fetchall()

        fading = []
        for uri, cat, imp, acc, la in rows:
            ret = self._retention((now - la) / _DAY_SECONDS, imp, acc)
            if self._soft <= ret < retention_max:
                fading.append({"uri": uri, "category": cat, "importance": imp, "retention": ret})

        fading.sort(key=lambda x: x["retention"])
        return fading[:limit]

    def purge(self, uris: list[str]) -> None:
        if not uris:
            return

        with self._lock:
            ph = ",".join("?" * len(uris))
            self._conn.execute(f"""DELETE FROM memory_meta WHERE uri IN ({ph})""", uris)
            self._conn.commit()
