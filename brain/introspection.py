import time
import threading
import logger
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool
from memory import HybridMemory
from config import settings, build_introspection_instructions


log = logger.get(__name__)


class IntrospectionOutput(BaseModel):
    insights: str = Field(description="Relationship insights and memory maintenance summaries will be stored in long-term memory.")
    calibrations: list[str] = Field(default_factory=list, description="The list of personality calibration signals includes a specific behavioral correction suggestion for each signal.")


class IntrospectionLoop:
    _IDLE_THRESHOLD = 3
    _MAX_IDLE_TICKS = 8
    _MAX_ERROR_RETRIES = 5

    def __init__(self, memory: HybridMemory):
        self._memory = memory
        self._agent: Agent | None = None
        self._interval: float = settings.introspection_interval
        self._last_activity: float = 0.0
        self._cond = threading.Condition()
        self._activity_flag = False
        self._stopped = False
        self._started = False
        self._calibration_buffer: list[str] = []
        self._calibration_lock = threading.Lock()

    def start(self):
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify()

    def notify_activity(self):
        self._last_activity = time.time()
        with self._cond:
            self._activity_flag = True
            self._cond.notify()

    @property
    def calibrations(self) -> list[str]:
        with self._calibration_lock:
            return list(self._calibration_buffer)

    # 反思内循环：
    #   等待（可被用户活动打断）→ 评估活跃状态 → 收集种子 → 执行反思 → 吸收结果
    #   退避策略：用户活跃时保持基础间隔；空闲后指数退避；出错独立退避，活跃时重置一切
    def _loop(self):
        errors = 0
        idle_ticks = 0

        while not self._stopped:
            base = self._interval
            ticks = errors if errors > 0 else idle_ticks
            wait = base * (1 << ticks) if ticks > 0 else base

            # 退避等待：新消息或 stop() 均可唤醒
            with self._cond:
                self._activity_flag = False
                self._cond.wait_for(lambda: self._activity_flag or self._stopped, timeout=wait)

            if self._stopped:
                break

            # 活跃判定：超过阈值视为空闲，逐步增加退避；否则重置所有计数器
            if self._last_activity > 0:
                if time.time() - self._last_activity > base * self._IDLE_THRESHOLD:
                    idle_ticks = min(idle_ticks + 1, self._MAX_IDLE_TICKS)
                else:
                    idle_ticks = 0
                    errors = 0

            try:
                # 收集反思种子：情景片段 + 衰减记忆，为空则跳过不计入退避
                seed = self._memory.gather_introspection_seed()
                if seed.is_empty:
                    log.info("introspection skipped: seed empty")
                    continue

                log.info("introspection seed: %s", ", ".join(seed.section_names))
                result = Runner.run_sync(
                    starting_agent=self._get_agent(),
                    input=seed.render(),
                )

                # 反思结果写入长期记忆
                output = result.final_output  # IntrospectionOutput
                with self._calibration_lock:
                    self._calibration_buffer = output.calibrations[-5:]

                self._memory.absorb_introspection(output.insights)
                log.info("introspection done, insights=%s, calibrations=%d",
                         output.insights.replace("\n", ""), len(output.calibrations))
                errors = 0
            except Exception as e:
                errors = min(errors + 1, self._MAX_ERROR_RETRIES)
                log.warning("introspection error: %s (retry #%d)", e, errors, exc_info=True)

    def _get_agent(self) -> Agent:
        if self._agent:
            return self._agent

        memory = self._memory

        @function_tool
        def recall(query: str) -> str:
            """Search long-term memory by semantic similarity.

            Args:
                query: Natural language search query describing the topic or concept to look up.

            Returns:
                A newline-separated list of matching memory entries, each formatted as:
                "- [category | score: N.NN] abstract (uri: memory_uri)"
                Returns a "not found" message if no memories match.
            """

            results = memory.persona.search(query)
            if settings.memory_decay_enabled:
                results = memory.decay.apply_decay(results)
            if not results:
                log.info("introspection recall, query=%r -> empty", query)
                return "No relevant memories were found."

            output = "\n".join(f"\t- <{m.get('category') or 'uncategorized'} | score: {m['score']:.2f}> {m.get('abstract', '')} (uri: {m.get('uri', '')})" for m in results)

            log.info("introspection recall, query=%r -> %d hits", query, len(results))
            log.debug("recall results:\n%s", output)
            return output

        @function_tool
        def recall_detail(uri: str) -> str:
            """Retrieve the full overview of a specific memory entry.

            Args:
                uri: The memory URI obtained from recall results (e.g. "viking://memories/...").

            Returns:
                The complete overview text of the memory. Returns an error message if
                the URI is invalid or inaccessible.
            """

            try:
                overview = memory.persona.read_overview(uri)
            except Exception as e:
                log.warning("introspection recall_detail failed: %s, uri=%s", e, uri, exc_info=True)
                return "Failed to retrieve overview."

            if overview is None:
                memory.decay.purge([uri])
                log.info("introspection recall_detail: memory gone, purged metadata, uri=%s", uri)
                return "This memory no longer exists — it has already been forgotten."

            log.info("introspection recall_detail, uri=%s -> %d chars", uri, len(overview))
            return overview

        @function_tool
        def reinforce(uri: str) -> str:
            """Reinforce a memory to prevent it from decaying. Resets the decay timer
            and increments the access count, increasing its long-term stability.

            Args:
                uri: The memory URI to reinforce (e.g. "viking://memories/...").

            Returns:
                A confirmation message indicating the memory has been reinforced.
            """

            memory.decay.record_accesses([{"uri": uri}])

            log.info("introspection reinforce, uri=%s", uri)
            return "Memory reinforcement successful."

        @function_tool
        def forget(uri: str) -> str:
            """Permanently delete a memory from both the long-term store and decay metadata.
            This action is irreversible. Only use for memories that are outdated, incorrect,
            or no longer relevant.

            Args:
                uri: The memory URI to delete (e.g. "viking://memories/...").

            Returns:
                A confirmation message on success, or an error message if deletion failed.
            """

            try:
                memory.persona.delete_memory(uri)
            except Exception as e:
                log.warning("introspection forget delete failed: %s, uri=%s", e, uri, exc_info=True)
                return "The operation failed."

            memory.decay.purge([uri])
            log.info("introspection forget, uri=%s", uri)
            return "The memory has been forgotten."

        self._agent = Agent(
            name="Introspection Agent",
            model=settings.introspection_model,
            instructions=build_introspection_instructions,
            tools=[recall, recall_detail, reinforce, forget],
            output_type=IntrospectionOutput,
        )

        return self._agent
