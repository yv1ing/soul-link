import asyncio
import logger
from agents import Agent, Runner
from memory import HybridMemory
from brain.emotion import EmotionTracker
from brain.introspection import IntrospectionLoop
from config import settings, build_soul_instructions


log = logger.get(__name__)


class Brain:
    def __init__(self, session_id: str = "default"):
        self.memory = HybridMemory(session_id=session_id)
        self._soul_agent: Agent | None = None
        self._emotion = EmotionTracker()
        self._emotion_task: asyncio.Task | None = None
        self._introspection = IntrospectionLoop(self.memory)

    def close(self):
        self._introspection.stop()
        if self._emotion_task is not None:
            self._emotion_task.cancel()
        self.memory.close()

    def introspect(self):
        self._introspection.start()

    async def think(self, text_input: str):
        self._introspection.notify_activity()

        # 等待上一轮情绪分析完成
        if self._emotion_task is not None:
            try:
                await self._emotion_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.warning("emotion task error: %s", e)
            self._emotion_task = None

        # 快照当前情绪状态（来自上一轮的分析结果）
        self.memory.set_emotional_state(self._emotion)
        self.memory.set_calibrations(self._introspection.calibrations)

        # 后台启动情绪分析（注入对话上下文，当前情绪快照由 tracker 内部自动注入）
        emotion_context = await asyncio.to_thread(
            self.memory.get_emotion_context, settings.emotion_context_turns
        )
        self._emotion_task = asyncio.create_task(
            self._emotion.update(text_input, context=emotion_context)
        )

        result = await Runner.run(
            starting_agent=self._get_soul_agent(),
            input=text_input,
            session=self.memory,
        )
        return result.final_output

    def _get_soul_agent(self) -> Agent:
        if self._soul_agent:
            return self._soul_agent

        self._soul_agent = Agent(
            name="Soul Agent",
            model=settings.soul_model,
            instructions=build_soul_instructions,
        )

        return self._soul_agent
