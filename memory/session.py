import os
import json
import time
import sqlite3
import threading
from agents import TResponseInputItem
from config import settings
from memory.importance import score_importance


class SessionStore:
    def __init__(self, session_id: str, max_items: int | None = None):
        self.session_id = session_id
        self.max_items = max_items or settings.memory_max_history
        self._protected_count = max(1, int(self.max_items * settings.memory_protected_ratio))

        self._conn = sqlite3.connect(os.path.join(settings.data_path, "sessions.db"), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()

        self._init_schema()

    def close_db(self) -> None:
        self._conn.close()

    def _init_schema(self):
        with self._lock:
            self._conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS session_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        item_data TEXT NOT NULL,
                        importance REAL NOT NULL DEFAULT 0.0,
                        created_at REAL NOT NULL
                    )
                """
            )
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_si_sid ON session_items(session_id)
            """)

            columns = {r[1] for r in self._conn.execute("""PRAGMA table_info(session_items)""").fetchall()}
            if "importance" not in columns:
                self._conn.execute("""
                    ALTER TABLE session_items ADD COLUMN importance REAL NOT NULL DEFAULT 0.0
                """)

            self._conn.commit()

    def add(self, items: list[TResponseInputItem]) -> None:
        with self._lock:
            for item in items:
                role = str(item.get("role", ""))
                content = str(item.get("content", ""))
                if role and content:
                    self._conn.execute(
                        """
                            INSERT INTO session_items (session_id, item_data, importance, created_at) VALUES (?,?,?,?)
                        """,
                        (self.session_id, json.dumps(item, ensure_ascii=False), score_importance(role, content), time.time())
                    )
            self._evict()
            self._conn.commit()

    def get(self, limit: int | None = None) -> list[TResponseInputItem]:
        with self._lock:
            if limit is not None:
                rows = self._conn.execute(
                    """
                        SELECT item_data FROM (SELECT item_data, id FROM session_items WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC
                    """,
                    (self.session_id, limit)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT item_data FROM session_items WHERE session_id=? ORDER BY id ASC""",
                    (self.session_id,)
                ).fetchall()

            return [json.loads(r[0]) for r in rows]

    def pop(self) -> TResponseInputItem | None:
        with self._lock:
            row = self._conn.execute(
                """
                    SELECT id, item_data FROM session_items WHERE session_id=? ORDER BY id DESC LIMIT 1
                """,
                (self.session_id,)
            ).fetchone()

            if not row:
                return None

            self._conn.execute(
                """DELETE FROM session_items WHERE id=?""",
                (row[0],)
            )
            self._conn.commit()

            return json.loads(row[1])

    def clear(self) -> None:
        with self._lock:
            self._conn.execute(
                """DELETE FROM session_items WHERE session_id=?""",
                (self.session_id,)
            )
            self._conn.commit()

    def _evict(self):
        count = self._conn.execute(
            """SELECT COUNT(*) FROM session_items WHERE session_id=?""",
            (self.session_id,)
        ).fetchone()[0]

        if count <= self.max_items:
            return

        rows = self._conn.execute(
            """SELECT id, importance FROM session_items WHERE session_id=? ORDER BY id ASC""",
            (self.session_id,)
        ).fetchall()

        protected = {r[0] for r in rows[-self._protected_count:]}
        candidates = sorted([(rid, imp) for rid, imp in rows if rid not in protected], key=lambda x: (x[1], x[0]))
        to_delete = [rid for rid, _ in candidates[:count - self.max_items]]

        if to_delete:
            ph = ",".join("?" * len(to_delete))
            self._conn.execute(
                f"""DELETE FROM session_items WHERE id IN ({ph})""",
                to_delete
            )

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def lock(self) -> threading.Lock:
        return self._lock