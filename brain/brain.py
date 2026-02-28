import asyncio
import logger
from agents import Agent, Runner
from agents.mcp import MCPServerManager, MCPServerStdio
from memory import HybridMemory
from brain.emotion import EmotionTracker
from brain.introspection import IntrospectionLoop
from brain.skills import get_skill, refresh_skill_catalog, build_soul_instructions
from config import settings


log = logger.get(__name__)


class Brain:
    def __init__(self, session_id: str = "default"):
        mcp_servers = [
            MCPServerStdio(
                name=cfg.name,
                params={"command": cfg.command, "args": cfg.args, **({"env": cfg.env} if cfg.env else {})},
                require_approval=cfg.require_approval,
            )
            for cfg in settings.mcp_servers
        ]

        self.mcp_manager = MCPServerManager(mcp_servers)
        self.memory = HybridMemory(session_id=session_id)

        self._soul_agent: Agent | None = None
        self._emotion = EmotionTracker()
        self._emotion_task: asyncio.Task | None = None
        self._introspection = IntrospectionLoop(self.memory)

    async def start(self):
        await self.mcp_manager.connect_all()

    async def close(self):
        self._introspection.stop()
        if self._emotion_task is not None:
            self._emotion_task.cancel()
            try:
                await self._emotion_task
            except asyncio.CancelledError:
                pass
            self._emotion_task = None
        try:
            await self.mcp_manager.cleanup_all()
        except Exception as e:
            log.warning("mcp manager cleanup error: %s", e)
        await asyncio.to_thread(self._introspection.join)
        self.memory.close()

    async def think(self, text_input: str):
        self._introspection.notify_activity()
        await asyncio.to_thread(refresh_skill_catalog, settings.skills_path)

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

    def introspect(self):
        self._introspection.start()

    def _get_soul_agent(self) -> Agent:
        if self._soul_agent is None:
            self._soul_agent = Agent(
                name="Soul Agent",
                model=settings.soul_model,
                instructions=build_soul_instructions,
                mcp_servers=self.mcp_manager.active_servers,
                tools=[
                    get_skill,
                ],
            )
        else:
            self._soul_agent.mcp_servers = self.mcp_manager.active_servers

        return self._soul_agent
