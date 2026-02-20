import os
import threading
import openviking as ov
from openviking.message import TextPart
from config import settings


_CATEGORY_KEYWORDS = ("preferences", "entities", "events", "cases", "patterns", "profile")


def _category_from_uri(uri: str) -> str:
    for kw in _CATEGORY_KEYWORDS:
        if kw in uri:
            return kw

    return ""


class PersonaStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._lock = threading.Lock()
        self._message_count = 0

        self._viking = ov.SyncOpenViking(path=os.path.join(settings.data_path, "openviking"))
        self._viking.initialize()

        self._session = self._viking.session(session_id)
        self._session.load()
        self._session._messages.clear()

    def feed(self, role: str, content: str) -> None:
        with self._lock:
            self._session.add_message(role, [TextPart(text=content)])
            self._message_count += 1

    def commit(self) -> dict:
        with self._lock:
            result = self._session.commit()
            self._message_count = 0
            return result

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        with self._lock:
            result = self._viking.search(query, limit=limit or settings.memory_search_limit)
            return [
                {
                    "uri": m.uri,
                    "score": m.score,
                    "abstract": m.abstract,
                    "overview": m.overview,
                    "category": m.category or _category_from_uri(m.uri),
                }
                for m in result.memories
            ]

    def load_profile(self) -> str | None:
        with self._lock:
            try:
                content = self._viking.read("viking://user/memories/profile.md")
                return content if content and content.strip() else None
            except Exception:
                return None

    def read_overview(self, uri: str) -> str:
        with self._lock:
            return self._viking.read(uri)

    def delete_memory(self, uri: str) -> None:
        with self._lock:
            self._viking.rm(uri)

    @property
    def message_count(self) -> int:
        return self._message_count
