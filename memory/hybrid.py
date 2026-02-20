import asyncio
from agents import SessionABC, TResponseInputItem
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        self.session = SessionStore(session_id=session_id, max_items=settings.memory_max_history)
        self.decay = MemoryDecay(conn=self.session.conn, lock=self.session.lock)
        self.episodic = EpisodicBuffer(session_id=session_id, conn=self.session.conn, lock=self.session.lock)
        self.persona = PersonaStore(session_id=session_id)

        self._pending_messages: list[dict] = []
        self._profile_cache: str | None = None
        self._profile_cached = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self.persona.message_count > 0:
                self.persona.commit()

            pending = self._flush_pending()
            if pending:
                self.episodic.add(summary=pending[0], message_count=pending[1])
        finally:
            self.session.close_db()

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        if self._closed:
            return []

        prefix: list[TResponseInputItem] = []

        profile, items = await asyncio.gather(
            asyncio.to_thread(self._load_profile_cached),
            asyncio.to_thread(self.session.get, limit),
        )

        if profile:
            prefix.append({"role": "system", "content": f"{_PROFILE_TAG}\nThe following is user profile information:\n{profile}"})

        query = self._build_query(items)
        if query:
            memories, episodes = await asyncio.gather(
                asyncio.to_thread(self.persona.search, query),
                asyncio.to_thread(self.episodic.get_recent),
            )

            if settings.memory_decay_enabled:
                memories = self.decay.apply_decay(memories)

            ltm_block = self._build_ltm_block(memories, episodes)
            if ltm_block:
                prefix.append({"role": "system", "content": ltm_block})

            if memories and settings.memory_decay_enabled:
                self.decay.record_accesses(memories)

        return prefix + items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if self._closed:
            return

        filtered = [it for it in items
                    if not str(it.get("content", "")).startswith((_LTM_TAG, _PROFILE_TAG))]
        if not filtered:
            return

        self.session.add(filtered)

        has_urgent = False
        for it in filtered:
            role = str(it.get("role", ""))
            content = str(it.get("content", ""))
            if role in ("user", "assistant") and content:
                self.persona.feed(role, content)
                self._pending_messages.append({"role": role, "text": content})

                if role == "user" and score_importance(role, content) >= settings.memory_urgent_importance:
                    has_urgent = True

        count = self.persona.message_count
        if count >= settings.memory_commit_threshold or (has_urgent and count >= 1):
            self._safe_commit()

    async def pop_item(self) -> TResponseInputItem | None:
        return self.session.pop()

    async def clear_session(self) -> None:
        self.session.clear()

    def _flush_pending(self) -> tuple[str, int] | None:
        if not self._pending_messages:
            return None
        summary = self._summarize_episode(self._pending_messages)
        count = len(self._pending_messages)
        self._pending_messages.clear()

        return (summary, count) if summary else None

    def _load_profile_cached(self) -> str | None:
        if not self._profile_cached:
            self._profile_cache = self.persona.load_profile()
            self._profile_cached = True
        return self._profile_cache

    def _safe_commit(self) -> None:
        self.persona.commit()
        self._profile_cached = False

        pending = self._flush_pending()
        if pending:
            self.episodic.add(summary=pending[0], message_count=pending[1])

        if settings.memory_decay_enabled:
            forgotten = self.decay.collect_forgotten()
            if forgotten:
                deleted = []
                for item in forgotten:
                    try:
                        self.persona.delete_memory(item["uri"])
                        deleted.append(item["uri"])
                    except Exception:
                        pass
                if deleted:
                    self.decay.purge(deleted)

    def _build_ltm_block(self, memories: list[dict], episodes: list[dict]) -> str:
        high_th = settings.memory_high_score_threshold
        low_th = settings.memory_low_score_threshold

        high_candidates: list[tuple[dict, str]] = []
        medium_items: list[str] = []
        uris_to_fetch: list[tuple[int, str]] = []

        for m in memories:
            score = m.get("score", 0)
            cat = m.get("category", "")

            if score >= high_th:
                overview = m.get("overview")
                abstract = m.get("abstract", "")
                if not overview and not abstract:
                    continue
                idx = len(high_candidates)
                high_candidates.append((m, overview or abstract))
                if not overview and m.get("uri"):
                    uris_to_fetch.append((idx, m["uri"]))
            elif score >= low_th:
                abstract = m.get("abstract", "")
                if abstract:
                    medium_items.append(f"- [{cat}] {abstract}")

        if uris_to_fetch:
            with ThreadPoolExecutor(max_workers=min(len(uris_to_fetch), 4)) as pool:
                futures = {pool.submit(self.persona.read_overview, uri): idx for idx, uri in uris_to_fetch}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        if isinstance(result, str) and result:
                            high_candidates[idx] = (high_candidates[idx][0], result)
                    except Exception:
                        pass

        high_items = [f"- [{m.get('category', '')}] {text}" for m, text in high_candidates]

        sections: list[str] = []
        if high_items:
            sections.append("## Highly associative memories\n" + "\n".join(high_items))
        if medium_items:
            sections.append("## Other reference memories\n" + "\n".join(medium_items))
        if episodes:
            ep_lines = [f"- {ep['summary']}" for ep in episodes[-3:] if ep.get("summary")]
            if ep_lines:
                sections.append("## Recent conversation context\n" + "\n".join(ep_lines))

        if not sections:
            return ""

        return f"{_LTM_TAG}\nThe following information comes from the long-term memory system and is used to maintain cognitive coherence:\n" + "\n\n".join(sections)

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
        pairs: list[str] = []
        last_user: str | None = None

        for m in messages:
            text = m.get("text", "")
            if not text:
                continue
            if m.get("role") == "user":
                if last_user is not None:
                    pairs.append(f"Q: {last_user}")
                last_user = text[:200]
            elif m.get("role") == "assistant" and last_user is not None:
                pairs.append(f"Q: {last_user} → A: {text[:200]}")
                last_user = None

        if last_user is not None:
            pairs.append(f"Q: {last_user}")

        return " | ".join(pairs[:4]) if pairs else ""
