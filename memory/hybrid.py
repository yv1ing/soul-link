import os
import time
import sqlite3
import asyncio
import threading
from agents import SessionABC, TResponseInputItem
from config import settings
from memory.session import SessionStore
from memory.persona import PersonaStore
from memory.episodic import EpisodicBuffer
from memory.decay import MemoryDecay
from memory.importance import score_importance


_LTM_TAG = "[LONG_TERM_MEMORY]"
_PROFILE_TAG = "[USER_PROFILE]"


class HybridMemory(SessionABC):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._closed = False

        self._db_conn = sqlite3.connect(os.path.join(settings.data_path, "sessions.db"), check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_lock = threading.Lock()

        self.decay = MemoryDecay(conn=self._db_conn, lock=self._db_lock)
        self.episodic = EpisodicBuffer(session_id=session_id, conn=self._db_conn, lock=self._db_lock)
        self.session = SessionStore(session_id=session_id, max_items=settings.memory_max_history, conn=self._db_conn, lock=self._db_lock)
        self.persona = PersonaStore(session_id=session_id)

        self._pending_messages: list[dict] = []

    # ------------------------------------------------------------------ #
    #  SessionABC 接口实现
    # ------------------------------------------------------------------ #

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        if self._closed:
            return []

        prefix_blocks: list[TResponseInputItem] = []

        profile = await self.persona.load_profile()
        if profile:
            prefix_blocks.append({
                "role": "system",
                "content": f"{_PROFILE_TAG}\n以下是关于用户的画像信息：\n{profile}",
            })

        items = await self.session.get(limit=limit)

        query = self._build_query(items)
        if query:
            memories, episodes = await asyncio.gather(
                self.persona.search(query),
                self.episodic.get_recent(),
            )

            if settings.memory_decay_enabled:
                memories = await self.decay.apply_decay(memories)

            ltm_block = await self._build_ltm_block(memories, episodes)
            if ltm_block:
                prefix_blocks.append({
                    "role": "system",
                    "content": ltm_block,
                })

            if memories and settings.memory_decay_enabled:
                await self.decay.record_accesses(memories)

        return prefix_blocks + items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if self._closed:
            return

        filtered = [
            it for it in items
            if not self._is_injected_block(it)
        ]
        if not filtered:
            return

        await self.session.add(filtered)

        has_urgent = False
        for it in filtered:
            role = str(it.get("role", ""))
            content = str(it.get("content", ""))
            if role in ("user", "assistant") and content:
                await self.persona.feed_message(role, content)
                self._pending_messages.append({"role": role, "text": content})

                if role == "user":
                    imp = score_importance(role, content)
                    if imp >= settings.memory_urgent_importance:
                        has_urgent = True

        await self._maybe_commit(urgent=has_urgent)

    async def pop_item(self) -> TResponseInputItem | None:
        return await self.session.pop()

    async def clear_session(self) -> None:
        await self.session.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self.persona.message_count > 0:
                await self.persona.commit()

            if self._pending_messages:
                summary = self._summarize_episode(self._pending_messages)
                if summary:
                    await self.episodic.add(
                        summary=summary,
                        message_count=len(self._pending_messages),
                    )
                self._pending_messages.clear()
        finally:
            self._db_conn.close()

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            asyncio.run(self.close())
        except RuntimeError:
            self._sync_close_fallback()

    # ------------------------------------------------------------------ #
    #  内部方法
    # ------------------------------------------------------------------ #

    def _sync_close_fallback(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self.persona.message_count > 0:
                self.persona.sync_commit()

            if self._pending_messages:
                summary = self._summarize_episode(self._pending_messages)
                if summary:
                    with self._db_lock:
                        self._db_conn.execute(
                            "INSERT INTO episodic_buffer (session_id, summary, message_count, created_at) VALUES (?,?,?,?)",
                            (self.session_id, summary, len(self._pending_messages), time.time()),
                        )
                        self._db_conn.commit()
                self._pending_messages.clear()
        finally:
            self._db_conn.close()

    async def _maybe_commit(self, urgent: bool = False) -> None:
        msg_count = self.persona.message_count
        should_commit = (
            msg_count >= settings.memory_commit_threshold
            or (urgent and msg_count >= 1)
        )

        if should_commit:
            await self._safe_commit()

    async def _safe_commit(self) -> None:
        await self.persona.commit()

        pending = self._pending_messages.copy()
        self._pending_messages.clear()

        summary = self._summarize_episode(pending)
        if summary:
            await self.episodic.add(
                summary=summary,
                message_count=len(pending),
            )

        if settings.memory_decay_enabled:
            await self._cleanup_forgotten()

    async def _cleanup_forgotten(self) -> None:
        forgotten = await self.decay.collect_forgotten()
        if not forgotten:
            return

        deleted_uris: list[str] = []
        for item in forgotten:
            uri = item["uri"]
            try:
                await self.persona.delete_memory(uri)
                deleted_uris.append(uri)
            except Exception:
                pass

        if deleted_uris:
            await self.decay.purge(deleted_uris)

    async def _build_ltm_block(self, memories: list[dict], episodes: list[dict]) -> str:
        header = f"{_LTM_TAG}\n以下信息来自长期记忆系统，用于保持认知连贯性：\n"

        high_threshold = settings.memory_high_score_threshold
        low_threshold = settings.memory_low_score_threshold

        # 分类记忆并收集需要拉取overview的URI
        high_candidates: list[tuple[dict, str]] = []
        medium_items: list[str] = []
        uris_to_fetch: list[tuple[int, str]] = []

        for m in memories:
            score = m.get("score", 0)
            category = m.get("category", "")

            if score >= high_threshold:
                overview = m.get("overview")
                abstract = m.get("abstract", "")

                if not overview and not abstract:
                    continue

                idx = len(high_candidates)
                high_candidates.append((m, overview or abstract))

                if not overview and m.get("uri"):
                    uris_to_fetch.append((idx, m["uri"]))

            elif score >= low_threshold:
                abstract = m.get("abstract", "")
                if abstract:
                    medium_items.append(f"- [{category}] {abstract}")

        # 并行拉取缺失的overview
        if uris_to_fetch:
            tasks = [self.persona.read_overview(uri) for _, uri in uris_to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (idx, _), result in zip(uris_to_fetch, results):
                if isinstance(result, str) and result:
                    high_candidates[idx] = (high_candidates[idx][0], result)

        high_items: list[str] = []
        for m, text in high_candidates:
            category = m.get("category", "")
            high_items.append(f"- [{category}] {text}")

        # 组装最终注入块
        sections: list[str] = []
        if high_items:
            sections.append("## 高度相关记忆\n" + "\n".join(high_items))
        if medium_items:
            sections.append("## 其它参考记忆\n" + "\n".join(medium_items))

        if episodes:
            ep_lines = [f"- {ep['summary']}" for ep in episodes[-3:] if ep.get("summary")]
            if ep_lines:
                sections.append("## 近期会话脉络\n" + "\n".join(ep_lines))

        if not sections:
            return ""

        return header + "\n\n".join(sections)

    @staticmethod
    def _build_query(items: list[TResponseInputItem]) -> str:
        user_msgs: list[str] = []
        last_assistant: str | None = None

        for it in reversed(items):
            content = it.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            role = it.get("role")
            if role == "user" and len(user_msgs) < 3:
                user_msgs.append(content)
            elif role == "assistant" and last_assistant is None:
                last_assistant = content

            if len(user_msgs) >= 3:
                break

        if not user_msgs:
            return ""

        parts = list(reversed(user_msgs))
        if last_assistant and len(last_assistant) < 200:
            parts.append(last_assistant)

        return "\n".join(parts)

    @staticmethod
    def _summarize_episode(messages: list[dict]) -> str:
        """从一批消息中生成 Q→A 配对式摘要"""
        pairs: list[str] = []
        last_user: str | None = None

        for m in messages:
            role = m.get("role", "")
            text = m.get("text", "")
            if not text:
                continue

            if role == "user":
                if last_user is not None:
                    pairs.append(f"Q: {last_user}")
                last_user = text[:200]
            elif role == "assistant" and last_user is not None:
                pairs.append(f"Q: {last_user} → A: {text[:200]}")
                last_user = None

        if last_user is not None:
            pairs.append(f"Q: {last_user}")

        if not pairs:
            return ""

        return " | ".join(pairs[:4])

    @staticmethod
    def _is_injected_block(item: TResponseInputItem) -> bool:
        content = str(item.get("content", ""))
        return content.startswith(_LTM_TAG) or content.startswith(_PROFILE_TAG)
