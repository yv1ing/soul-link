import json
import time
import asyncio
import sqlite3
import threading
from agents import TResponseInputItem
from config import settings
from memory.importance import score_importance


class SessionStore:
    def __init__(self, session_id: str, max_items: int, conn: sqlite3.Connection, lock: threading.Lock):
        self.session_id = session_id
        self.max_items = max_items or settings.memory_max_history
        self._protected_count = max(1, int(self.max_items * settings.memory_protected_ratio))

        self._conn = conn
        self._lock = lock

        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    item_data TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_si_sid ON session_items(session_id)")

            cursor = self._conn.execute("PRAGMA table_info(session_items)")
            columns = {row[1] for row in cursor.fetchall()}
            if "importance" not in columns:
                self._conn.execute("ALTER TABLE session_items ADD COLUMN importance REAL NOT NULL DEFAULT 0.0")

            self._conn.commit()

    async def add(self, items: list[TResponseInputItem]) -> None:
        def _do():
            with self._lock:
                for item in items:
                    role = str(item.get("role", ""))
                    content = str(item.get("content", ""))

                    if role and content:
                        imp = score_importance(role, content)
                        self._conn.execute(
                            "INSERT INTO session_items (session_id, item_data, importance, created_at) VALUES (?,?,?,?)",
                            (self.session_id, json.dumps(item, ensure_ascii=False), imp, time.time()),
                        )

                self._evict()
                self._conn.commit()

        await asyncio.to_thread(_do)

    async def get(self, limit: int | None = None) -> list[TResponseInputItem]:
        def _do():
            with self._lock:
                if limit is not None:
                    rows = self._conn.execute(
                        "SELECT item_data FROM (SELECT item_data, id FROM session_items WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                        (self.session_id, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT item_data FROM session_items WHERE session_id=? ORDER BY id ASC",
                        (self.session_id,),
                    ).fetchall()

                return [json.loads(r[0]) for r in rows]

        return await asyncio.to_thread(_do)

    async def pop(self) -> TResponseInputItem | None:
        def _do():
            with self._lock:
                row = self._conn.execute(
                    "SELECT id, item_data FROM session_items WHERE session_id=? ORDER BY id DESC LIMIT 1",
                    (self.session_id,),
                ).fetchone()

                if not row:
                    return None

                self._conn.execute("DELETE FROM session_items WHERE id=?", (row[0],))
                self._conn.commit()
                return json.loads(row[1])

        return await asyncio.to_thread(_do)

    async def clear(self) -> None:
        def _do():
            with self._lock:
                self._conn.execute("DELETE FROM session_items WHERE session_id=?", (self.session_id,))
                self._conn.commit()

        await asyncio.to_thread(_do)

    def _evict(self):
        count = self._conn.execute(
            "SELECT COUNT(*) FROM session_items WHERE session_id=?",
            (self.session_id,),
        ).fetchone()[0]

        if count <= self.max_items:
            return

        n_to_delete = count - self.max_items

        all_rows = self._conn.execute(
            "SELECT id, importance FROM session_items WHERE session_id=? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()

        protected_ids = {row[0] for row in all_rows[-self._protected_count:]}

        candidates = [(row_id, imp) for row_id, imp in all_rows if row_id not in protected_ids]
        candidates.sort(key=lambda x: (x[1], x[0]))

        to_delete = [row_id for row_id, _ in candidates[:n_to_delete]]

        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self._conn.execute(f"DELETE FROM session_items WHERE id IN ({placeholders})", to_delete)
