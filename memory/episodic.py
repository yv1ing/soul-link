import time
import sqlite3
import threading
from config import settings


class EpisodicBuffer:
    def __init__(self, session_id: str, conn: sqlite3.Connection, lock: threading.Lock):
        self.session_id = session_id
        self._max_episodes = settings.memory_episodic_limit
        self._conn = conn
        self._lock = lock
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL)
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eb_sid ON episodic_buffer(session_id)
            """)
            self._conn.commit()

    def add(self, summary: str, message_count: int = 0) -> None:
        with self._lock:
            self._conn.execute(
                """
                    INSERT INTO episodic_buffer (session_id, summary, message_count, created_at) VALUES (?,?,?,?)
                """,
                (self.session_id, summary, message_count, time.time())
            )

            count = self._conn.execute(
                """
                    SELECT COUNT(*) FROM episodic_buffer WHERE session_id=?
                """,
                (self.session_id,)
            ).fetchone()[0]

            if count > self._max_episodes:
                self._conn.execute(
                    """
                        DELETE FROM episodic_buffer WHERE id IN (SELECT id FROM episodic_buffer WHERE session_id=? ORDER BY id ASC LIMIT ?)
                    """,
                    (self.session_id, count - self._max_episodes)
                )

            self._conn.commit()

    def get_recent(self, limit: int | None = None) -> list[dict]:
        limit = limit or self._max_episodes
        with self._lock:
            rows = self._conn.execute(
                """
                    SELECT summary, message_count, created_at FROM episodic_buffer WHERE session_id=? ORDER BY id DESC LIMIT ?
                """,
                (self.session_id, limit)
            ).fetchall()

        return [{"summary": r[0], "message_count": r[1], "created_at": r[2]} for r in reversed(rows)]

