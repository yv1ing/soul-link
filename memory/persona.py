import os
import openviking as ov
from openviking.message import TextPart
from config import settings


class PersonaStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._message_count = 0

        self._viking = ov.SyncOpenViking(path=os.path.join(settings.data_path, "openviking"))
        self._viking.initialize()

        self._session = self._viking.session(session_id)
        self._session.load()
        # Discard loaded messages — already persisted, avoid expensive re-embedding
        self._session._messages.clear()

    def feed(self, role: str, content: str) -> None:
        self._session.add_message(role, [TextPart(text=content)])
        self._message_count += 1

    def commit(self) -> dict:
        result = self._session.commit()
        self._message_count = 0
        return result

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        result = self._viking.search(query, limit=limit or settings.memory_search_limit)
        return [{"abstract": m.abstract, "overview": m.overview, "score": m.score, "uri": m.uri, "category": m.category} for m in result.memories]

    def load_profile(self) -> str | None:
        try:
            content = self._viking.read("viking://user/memories/profile.md")
            return content if content and content.strip() else None
        except Exception:
            return None

    def read_overview(self, uri: str) -> str:
        return self._viking.overview(uri)

    def delete_memory(self, uri: str) -> None:
        self._viking.rm(uri)

    @property
    def message_count(self) -> int:
        return self._message_count
