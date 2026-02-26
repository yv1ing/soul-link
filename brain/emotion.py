import asyncio
import logger
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from agents import Agent, Runner
from config import settings


log = logger.get(__name__)

_EMA_NEW         = 0.6              # EMA 新值权重
_EMA_PREV        = 1.0 - _EMA_NEW   # EMA 前值权重，确保两者之和为 1
_SECONDARY_FADE  = 0.3              # 次情绪每轮向 blend_ratio=1.0 靠近的比例
_SECONDARY_CLEAR = 0.95             # blend_ratio 超过此值时清除次情绪


class PADDimensions(BaseModel):
    valence:     float = Field(description="Emotional valence, from -1 (extremely negative) to +1 (extremely positive).")
    arousal:     float = Field(description="Emotional arousal level, from 0 (calm) to 1 (excited).")
    dominance:   float = Field(description="Dominance level, from 0 (passive, vulnerable) to 1 (confident, in control).")
    description: str   = Field(description="A one-sentence natural language description of this emotional state, using '用户' as subject.")


class EmotionAnalysis(BaseModel):
    primary:     PADDimensions        = Field(description="The most prominent emotional state.")
    secondary:   PADDimensions | None = Field(default=None, description="Co-existing secondary emotion. Set to null if emotion is pure.")
    blend_ratio: float                = Field(default=1.0, description="Weight of primary emotion, 0.5~1.0. Use 1.0 when no secondary emotion.")


@dataclass
class PADState:
    valence:     float = 0.0   # -1 ~ +1
    arousal:     float = 0.5   # 0 ~ 1，默认中性
    dominance:   float = 0.5   # 0 ~ 1，默认中性
    description: str   = ""


@dataclass
class EmotionalState:
    primary:     PADState        = field(default_factory=PADState)
    secondary:   PADState | None = None
    blend_ratio: float           = 1.0          # 主情绪权重，1.0 表示无次情绪
    trend:       str             = "stable"     # rising / stable / falling


class EmotionTracker:
    _TREND_WINDOW    = 8
    _TREND_THRESHOLD = 0.08  # 判定 rising/falling 的最小 valence 差值
    _DECAY           = 0.85  # LLM 失败时 PAD 维度的衰减系数

    _RANGE_V = (-1.0, 1.0)
    _RANGE_A = (0.0,  1.0)
    _RANGE_D = (0.0,  1.0)

    def __init__(self) -> None:
        self._state   = EmotionalState()
        self._history: list[float] = []
        self._agent:   Agent | None = None

    def render(self) -> str:
        s, p = self._state, self._state.primary
        desc = f" ({p.description})" if p.description else ""
        lines = [
            f"primary: valence={p.valence:+.2f} arousal={p.arousal:.2f} dominance={p.dominance:.2f}{desc}",
            f"trend: {s.trend}",
        ]
        if s.secondary is not None and s.blend_ratio < 1.0:
            sec = s.secondary
            sec_desc = f" ({sec.description})" if sec.description else ""
            lines.append(
                f"secondary: valence={sec.valence:+.2f} arousal={sec.arousal:.2f}"
                f" dominance={sec.dominance:.2f}{sec_desc} (blend={s.blend_ratio:.2f})"
            )
        return "\n".join(lines)

    async def update(self, text: str, context: str = "") -> None:
        analysis = await self._analyze(text, context)
        if analysis is not None:
            self._apply_primary(analysis.primary)
            self._apply_secondary(analysis.secondary, analysis.blend_ratio)
            self._update_trend()

            pri = self._state.primary
            msg = "emotion update: primary(v=%+.2f a=%.2f d=%.2f desc=%s) trend=%s"
            args = [pri.valence, pri.arousal, pri.dominance, pri.description or "-", self._state.trend]
            if self._state.secondary is not None:
                sec = self._state.secondary
                msg += " | secondary(v=%+.2f a=%.2f d=%.2f desc=%s blend=%.2f)"
                args += [sec.valence, sec.arousal, sec.dominance, sec.description or "-", self._state.blend_ratio]
            log.info(msg, *args)
        else:
            self._decay_fallback()

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _ema_pad(self, current: PADState, new: PADDimensions) -> None:
        current.valence   = self._clamp(current.valence   * _EMA_PREV + new.valence   * _EMA_NEW, *self._RANGE_V)
        current.arousal   = self._clamp(current.arousal   * _EMA_PREV + new.arousal   * _EMA_NEW, *self._RANGE_A)
        current.dominance = self._clamp(current.dominance * _EMA_PREV + new.dominance * _EMA_NEW, *self._RANGE_D)
        current.description = new.description

    def _apply_primary(self, p: PADDimensions) -> None:
        self._ema_pad(self._state.primary, p)

    def _apply_secondary(self, sec: PADDimensions | None, blend_ratio: float) -> None:
        if sec is not None:
            ratio = self._clamp(blend_ratio, 0.5, 1.0)
            if self._state.secondary is None:
                self._state.secondary = PADState(
                    valence=self._clamp(sec.valence, *self._RANGE_V),
                    arousal=self._clamp(sec.arousal, *self._RANGE_A),
                    dominance=self._clamp(sec.dominance, *self._RANGE_D),
                    description=sec.description,
                )
                self._state.blend_ratio = ratio
            else:
                self._ema_pad(self._state.secondary, sec)
                self._state.blend_ratio = self._state.blend_ratio * _EMA_PREV + ratio * _EMA_NEW
        else:
            self._fade_secondary()

    def _fade_secondary(self) -> None:
        if self._state.secondary is None:
            return
        self._state.blend_ratio = min(
            1.0,
            self._state.blend_ratio + (1.0 - self._state.blend_ratio) * _SECONDARY_FADE,
        )
        if self._state.blend_ratio >= _SECONDARY_CLEAR:
            self._state.secondary = None
            self._state.blend_ratio = 1.0

    def _decay_fallback(self) -> None:
        # LLM 分析失败，PAD 向中性衰减
        s = self._state.primary
        s.valence   *= self._DECAY
        s.arousal    = 0.5 + (s.arousal - 0.5) * self._DECAY
        s.dominance  = 0.5 + (s.dominance - 0.5) * self._DECAY
        s.description = ""
        self._fade_secondary()
        self._update_trend()

    def _update_trend(self) -> None:
        self._history.append(self._state.primary.valence)
        if len(self._history) > self._TREND_WINDOW:
            self._history = self._history[-self._TREND_WINDOW:]
        self._state.trend = self._compute_trend()

    def _compute_trend(self) -> str:
        h = self._history
        if len(h) < 3:
            return "stable"
        mid = len(h) // 2
        diff = sum(h[mid:]) / (len(h) - mid) - sum(h[:mid]) / mid
        if diff > self._TREND_THRESHOLD:
            return "rising"
        if diff < -self._TREND_THRESHOLD:
            return "falling"
        return "stable"

    async def _analyze(self, text: str, context: str = "") -> EmotionAnalysis | None:
        parts = []
        if context:
            parts.append(f"[Recent conversation]\n{context}")
        parts.append(f"[Current emotional state]\n{self.render()}")
        parts.append(f"[Current user message]\n{text}")

        try:
            result = await Runner.run(
                starting_agent=self._get_agent(),
                input="\n\n".join(parts),
            )
            return result.final_output
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("emotion analysis failed: %s", e)
            return None

    def _get_agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(
                name="Emotion Analyzer",
                model=settings.emotion_model,
                instructions=settings.emotion_prompt,
                output_type=EmotionAnalysis,
            )
        return self._agent
