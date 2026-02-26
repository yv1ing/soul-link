import asyncio
import logger
from dataclasses import dataclass
from pydantic import BaseModel, Field
from agents import Agent, Runner
from config import settings


log = logger.get(__name__)


@dataclass
class EmotionalState:
    valence: float = 0.0      # -1 ~ +1（负面 ~ 正面）
    arousal: float = 0.0      # 0 ~ 1（平静 ~ 激动）
    dominance: float = 0.5    # 0 ~ 1（被动/脆弱 ~ 自信/掌控）
    trend: str = "stable"     # rising / stable / falling
    description: str = ""     # LLM 生成的自然语言描述


class EmotionAnalysis(BaseModel):
    valence: float = Field(description="Emotional valence, from -1 (extremely negative) to +1 (extremely positive).")
    arousal: float = Field(description="Emotional arousal level, from 0 (calm) to 1 (excited).")
    dominance: float = Field(description="Dominance level, from 0 (passive, vulnerable, helpless) to 1 (confident, in control, assertive).")
    description: str = Field(description="A one-sentence natural language description of the current emotional state.")


class EmotionTracker:
    _WINDOW = 8
    _DECAY = 0.85
    _CLAMP_V = (-1.0, 1.0)
    _CLAMP_A = (0.0, 1.0)
    _CLAMP_D = (0.0, 1.0)

    def __init__(self):
        self._state = EmotionalState()
        self._history: list[float] = []
        self._agent: Agent | None = None

    async def update(self, text: str, context: str = "", emotional_state: str = "") -> None:
        analysis = await self._analyze(text, context, emotional_state)

        if analysis is not None:
            valence = analysis.valence
            arousal = analysis.arousal
            dominance = analysis.dominance
            description = analysis.description
        else:
            self._state.valence *= self._DECAY
            self._state.arousal *= self._DECAY
            self._state.dominance = 0.5 + (self._state.dominance - 0.5) * self._DECAY
            self._state.description = ""
            self._update_trend()
            return

        self._state.valence = self._state.valence * 0.4 + valence * 0.6
        self._state.arousal = self._state.arousal * 0.4 + arousal * 0.6
        self._state.dominance = self._state.dominance * 0.4 + dominance * 0.6
        self._state.description = description

        self._state.valence = max(self._CLAMP_V[0], min(self._CLAMP_V[1], self._state.valence))
        self._state.arousal = max(self._CLAMP_A[0], min(self._CLAMP_A[1], self._state.arousal))
        self._state.dominance = max(self._CLAMP_D[0], min(self._CLAMP_D[1], self._state.dominance))

        self._update_trend()
        log.info("emotion update: v=%.2f a=%.2f d=%.2f trend=%s desc=%s", self._state.valence, self._state.arousal, self._state.dominance, self._state.trend, description[:50] if description else "(fallback)")

    async def _analyze(self, text: str, context: str = "", emotional_state: str = "") -> EmotionAnalysis | None:
        parts = []
        if context:
            parts.append(f"[Recent conversation]\n{context}")
        if emotional_state:
            parts.append(f"[Current emotional state]\n{emotional_state}")
        parts.append(f"[Current user message]\n{text}")
        rich_input = "\n\n".join(parts)

        try:
            result = await Runner.run(
                starting_agent=self._get_agent(),
                input=rich_input,
            )
            return result.final_output
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("emotion analysis failed: %s", e)
            return None

    def _get_agent(self) -> Agent:
        if self._agent is not None:
            return self._agent
        model = settings.emotion_model
        self._agent = Agent(
            name="Emotion Analyzer",
            model=model,
            instructions=settings.emotion_prompt,
            output_type=EmotionAnalysis,
        )
        return self._agent

    def _update_trend(self) -> None:
        self._history.append(self._state.valence)
        if len(self._history) > self._WINDOW:
            self._history = self._history[-self._WINDOW:]
        self._state.trend = self._compute_trend()

    def _compute_trend(self) -> str:
        h = self._history
        if len(h) < 3:
            return "stable"
        mid = len(h) // 2
        first_half = sum(h[:mid]) / mid
        second_half = sum(h[mid:]) / (len(h) - mid)
        diff = second_half - first_half
        if diff > 0.08:
            return "rising"
        elif diff < -0.08:
            return "falling"
        return "stable"

    def render(self) -> str:
        s = self._state
        desc = f" ({s.description})" if s.description else ""
        return f"valence={s.valence:+.2f} arousal={s.arousal:.2f} dominance={s.dominance:.2f} trend={s.trend}{desc}"

