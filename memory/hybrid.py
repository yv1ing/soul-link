import time
import logger
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from agents import SessionABC, TResponseInputItem
from config import settings
from memory.session import SessionStore
from memory.persona import PersonaStore
from memory.episodic import EpisodicBuffer
from memory.decay import MemoryDecay
from memory.importance import score_importance


log = logger.get(__name__)

_LTM_TAG = "[LONG_TERM_MEMORY]"
_PROFILE_TAG = "[USER_PROFILE]"
_EMOTIONAL_TAG = "[EMOTIONAL_CONTEXT]"
_CALIBRATION_TAG = "[CALIBRATION]"
_SELF_INTROSPECTION_TAG = "[SELF_INTROSPECTION]"


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


@dataclass
class IntrospectionSeed:
    profile: str | None = None
    emotional_state: str | None = None
    episodes: list[dict] = field(default_factory=list)
    fading: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.episodes and not self.fading

    @property
    def section_names(self) -> list[str]:
        names = []
        if self.profile:
            names.append("profile")
        if self.emotional_state:
            names.append("emotional_state")
        if self.episodes:
            names.append("episodes")
        if self.fading:
            names.append("fading")
        return names

    def render(self) -> str:
        sections: list[str] = []

        if self.profile:
            sections.append(f"## Current user profile\n{self.profile}")

        if self.emotional_state:
            sections.append(f"## Emotional state\n{self.emotional_state}")

        if self.episodes:
            lines = [f"- ({ep['time']}) {ep['summary']}" for ep in self.episodes]
            sections.append("## Summary of recent conversations\n" + "\n".join(lines))

        if self.fading:
            lines = []
            for m in self.fading:
                abstract = m.get("abstract", "")
                hint = f" {abstract}" if abstract else ""
                lines.append(
                    f"- [{m['category']} | retention: {m['retention']:.0%}]{hint} (uri: {m['uri']})"
                )
            sections.append("## Fading memory\n" + "\n".join(lines))

        return "\n\n".join(sections)


class HybridMemory(SessionABC):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._closed = False

        self.session = SessionStore(session_id=session_id, max_items=settings.memory_max_history)
        self.decay = MemoryDecay(conn=self.session.conn, lock=self.session.lock)
        self.episodic = EpisodicBuffer(session_id=session_id, conn=self.session.conn, lock=self.session.lock)
        self.persona = PersonaStore(session_id=session_id)

        self._pending_messages: list[dict] = []
        self._pending_lock = threading.Lock()
        self._commit_lock = threading.Lock()
        self._profile_lock = threading.Lock()
        self._profile_cache: str | None = None
        self._profile_cached = False
        self._emotional_state: str | None = None
        self._calibrations: list[str] = []

    def get_emotion_context(self, limit: int = 6) -> str:
        items = self.session.get(limit)
        parts = []
        for it in items:
            role = it.get("role", "")
            content = str(it.get("content", ""))
            if role in ("user", "assistant") and content:
                parts.append(f"[{role}]: {content[:200]}")
        return "\n".join(parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        with self._commit_lock:
            try:
                if self.persona.message_count > 0:
                    self.persona.commit()

                pending = self._flush_pending()
                if pending:
                    self.episodic.add(summary=pending[0], message_count=pending[1])
            finally:
                self.session.close_db()

    def set_emotional_state(self, tracker) -> None:
        self._emotional_state = tracker.render()

    def set_calibrations(self, calibrations: list[str]) -> None:
        self._calibrations = list(calibrations)

    def gather_introspection_seed(self) -> IntrospectionSeed:
        seed = IntrospectionSeed()

        seed.profile = self._load_profile_cached()
        seed.emotional_state = self._emotional_state

        episodes = self.episodic.get_recent()
        if episodes:
            seed.episodes = [
                {"summary": ep["summary"], "time": _format_ts(ep["created_at"])}
                for ep in episodes
                if ep.get("summary") and not ep["summary"].startswith(_SELF_INTROSPECTION_TAG)
            ]

        if not seed.episodes:
            pending_summary = self._peek_pending_summary()
            if pending_summary:
                seed.episodes = [
                    {"summary": pending_summary, "time": _format_ts(time.time())}
                ]

        if settings.memory_decay_enabled:
            fading = self.decay.get_fading()
            if fading:
                alive = []
                stale_uris = []
                for m in fading:
                    try:
                        overview = self.persona.read_overview(m["uri"])
                    except Exception:
                        alive.append(m)
                        continue
                    if overview is None:
                        stale_uris.append(m["uri"])
                        continue
                    m["abstract"] = overview[:100]
                    alive.append(m)
                if stale_uris:
                    self.decay.purge(stale_uris)
                    log.info("purged %d stale fading memory entries", len(stale_uris))
                seed.fading = alive

        return seed

    def absorb_introspection(self, introspection: str) -> None:
        if not introspection:
            return

        self.persona.feed("assistant", introspection)
        self.persona.commit()
        with self._profile_lock:
            self._profile_cached = False

        self.episodic.add(summary=f"{_SELF_INTROSPECTION_TAG} {introspection[:200]}", message_count=0)

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

        if self._emotional_state:
            prefix.append({"role": "system", "content": f"{_EMOTIONAL_TAG}\n{self._emotional_state}"})

        if self._calibrations:
            lines = "\n".join(f"- {c}" for c in self._calibrations)
            prefix.append({"role": "system", "content": f"{_CALIBRATION_TAG}\n近期自我反思中的注意事项：\n{lines}"})

        query = self._build_query(items)
        if query:
            memories, episodes = await asyncio.gather(
                asyncio.to_thread(self.persona.search, query),
                asyncio.to_thread(self.episodic.get_recent),
            )

            if settings.memory_decay_enabled:
                memories = await asyncio.to_thread(self.decay.apply_decay, memories)

            ltm_block = await asyncio.to_thread(self._build_ltm_block, memories, episodes)
            if ltm_block:
                prefix.append({"role": "system", "content": ltm_block})

            if memories and settings.memory_decay_enabled:
                await asyncio.to_thread(self.decay.record_accesses, memories)

        return prefix + items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if self._closed:
            return

        filtered = [it for it in items
                    if not str(it.get("content", "")).startswith((_LTM_TAG, _PROFILE_TAG, _EMOTIONAL_TAG, _CALIBRATION_TAG))]
        if not filtered:
            return

        await asyncio.to_thread(self.session.add, filtered)

        has_urgent = False
        pairs: list[tuple[str, str]] = []
        for it in filtered:
            role = str(it.get("role", ""))
            content = str(it.get("content", ""))
            if role in ("user", "assistant") and content:
                pairs.append((role, content))
                if role == "user" and score_importance(role, content) >= settings.memory_urgent_importance:
                    has_urgent = True

        if pairs:
            await asyncio.to_thread(self._ingest, pairs)

        count = self.persona.message_count
        if count >= settings.memory_commit_threshold or (has_urgent and count >= 1):
            asyncio.get_running_loop().run_in_executor(None, self._background_commit)

    async def pop_item(self) -> TResponseInputItem | None:
        return await asyncio.to_thread(self.session.pop)

    async def clear_session(self) -> None:
        await asyncio.to_thread(self.session.clear)

    def _ingest(self, pairs: list[tuple[str, str]]) -> None:
        for role, content in pairs:
            self.persona.feed(role, content)
        with self._pending_lock:
            self._pending_messages.extend({"role": role, "text": content} for role, content in pairs)

    def _background_commit(self) -> None:
        if self._closed:
            return
        with self._commit_lock:
            if self._closed:
                return
            try:
                self._safe_commit()
            except Exception as e:
                log.warning("background memory commit failed: %s", e, exc_info=True)

    def _flush_pending(self) -> tuple[str, int] | None:
        with self._pending_lock:
            if not self._pending_messages:
                return None
            messages = list(self._pending_messages)
            self._pending_messages.clear()
        summary = self._summarize_episode(messages)
        return (summary, len(messages)) if summary else None

    def _peek_pending_summary(self) -> str | None:
        with self._pending_lock:
            if not self._pending_messages:
                return None
            messages = list(self._pending_messages)
        return self._summarize_episode(messages) or None

    def _load_profile_cached(self) -> str | None:
        with self._profile_lock:
            if not self._profile_cached:
                self._profile_cache = self.persona.load_profile()
                self._profile_cached = True
            return self._profile_cache

    def _safe_commit(self) -> None:
        self.persona.commit()
        with self._profile_lock:
            self._profile_cached = False

        pending = self._flush_pending()
        if pending:
            self.episodic.add(summary=pending[0], message_count=pending[1])

        if settings.memory_decay_enabled:
            forgotten = self.decay.collect_forgotten()
            if forgotten:
                for item in forgotten:
                    try:
                        self.persona.delete_memory(item["uri"])
                    except Exception as e:
                        log.debug("failed to delete forgotten memory uri=%s, %s", item["uri"], e)
                self.decay.purge([item["uri"] for item in forgotten])

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

        for idx, uri in uris_to_fetch:
            try:
                overview = self.persona.read_overview(uri)
            except Exception:
                continue
            if overview:
                high_candidates[idx] = (high_candidates[idx][0], overview)

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
