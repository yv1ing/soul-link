import os
import asyncio
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

        # 提交上次运行残留的未提交消息，防止重启后重复提取记忆
        if self._session.messages:
            try:
                self._session.commit()
            except Exception:
                self._session._messages.clear()

    async def feed_message(self, role: str, content: str) -> None:
        """向缓冲区添加一条消息"""
        def _do():
            self._session.add_message(role, [TextPart(text=content)])

        await asyncio.to_thread(_do)
        self._message_count += 1

    async def commit(self) -> dict:
        """将缓冲区消息持久化为长期记忆"""
        def _do():
            return self._session.commit()

        result = await asyncio.to_thread(_do)
        self._message_count = 0
        return result

    def sync_commit(self) -> dict:
        result = self._session.commit()
        self._message_count = 0
        return result

    async def search(self, query: str, limit: int | None = None) -> list[dict]:
        """语义检索长期记忆"""
        def _do():
            result = self._viking.search(
                query,
                limit=limit or settings.memory_search_limit,
            )
            return [
                {
                    "abstract": m.abstract,
                    "overview": m.overview,
                    "score": m.score,
                    "uri": m.uri,
                    "category": m.category,
                }
                for m in result.memories
            ]

        return await asyncio.to_thread(_do)

    async def load_profile(self) -> str | None:
        """加载用户画像"""
        def _do():
            try:
                content = self._viking.read("viking://user/memories/profile.md")
                return content if content and content.strip() else None
            except Exception:
                return None

        return await asyncio.to_thread(_do)

    async def read_overview(self, uri: str) -> str:
        """读取记忆概览"""
        def _do():
            return self._viking.overview(uri)

        return await asyncio.to_thread(_do)

    async def delete_memory(self, uri: str) -> None:
        """删除记忆元素"""
        def _do():
            self._viking.rm(uri)

        await asyncio.to_thread(_do)

    @property
    def message_count(self) -> int:
        return self._message_count
