import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openviking_config_file: str = ""

    openai_base_url: str = ""
    openai_api_key: str = ""

    brain_model: str = ""
    brain_prompt: str

    data_path: str = "tdata"

    memory_max_history: int = 20        # 短期记忆最大保留条数
    memory_search_limit: int = 5        # 单次检索返回的长期记忆上限
    memory_commit_threshold: int = 10   # 短期记忆触发持久化提交阈值

    memory_protected_ratio: float = 0.6     # 淘汰时保护最近 N * ratio 条消息
    memory_urgent_importance: float = 0.7   # 重要性达到该值立即触发紧急持久化提交

    memory_high_score_threshold: float = 0.85   # 重要性 ≥ 此值注入为 L1（overview 级别）
    memory_low_score_threshold: float = 0.5     # 重要性 ≥ 此值注入为 L0（abstract 级别）

    memory_episodic_limit: int = 5                  # 情景缓冲区保留的最近消息数量
    memory_decay_enabled: bool = True               # 是否启用基于艾宾浩斯模型的记忆衰减机制
    memory_decay_base_stability: float = 30.0       # 基础稳定性（天），半衰期 ≈ base * ln(2)
    memory_decay_importance_weight: float = 1.5     # 重要成都对稳定性的影响权重
    memory_decay_access_weight: float = 1.0         # 访问次数对稳定性的影响权重
    memory_decay_soft_threshold: float = 0.15       # 保持率低于该值时在检索阶段过滤（软遗忘）
    memory_decay_hard_threshold: float = 0.02       # 保持率低于该值时在检索阶段删除（硬遗忘）

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

def clear_openviking_logger():
    for _name, _obj in logging.Logger.manager.loggerDict.items():
        if isinstance(_obj, logging.Logger) and _name.startswith(("openviking", "openviking_cli")):
            _obj.setLevel(logging.CRITICAL)
            for _h in _obj.handlers:
                _h.setLevel(logging.CRITICAL)

with open("prompts/SOUL.md", "r", encoding="utf-8") as f:
    brain_prompt = f.read().strip()

settings = Settings(brain_prompt=brain_prompt)

os.makedirs(settings.data_path, exist_ok=True)
