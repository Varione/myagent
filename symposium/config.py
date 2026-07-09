"""
配置模块 - 模型注册 + 研讨参数
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 模型配置
# ──────────────────────────────────────────────


@dataclass
class ModelConfig:
    """单个模型注册信息"""
    name: str                          # 模型标识名
    provider: str                      # "local" | "remote"
    endpoint: str                      # API 端点 URL
    api_key: str = ""                  # API Key（远程模型用）
    model_name: str = ""               # 实际模型名（空则用 name）
    temperature: float = 0.7
    max_tokens: int = 4096
    description: str = ""              # 模型能力描述（用于路由决策）
    weight: float = 1.0                # 投票/仲裁时的权重


@dataclass
class DeliberationConfig:
    """圆桌研讨参数"""
    num_rounds: int = 3                # 研讨轮数
    max_tokens_per_response: int = 2048
    temperature: float = 0.7


@dataclass
class ExecutionConfig:
    """执行阶段参数"""
    max_concurrent_tasks: int = 3      # 最大并行任务数
    progress_report_interval: float = 1.0  # 进度上报间隔（秒）


@dataclass
class SymposiumConfig:
    """全局配置"""
    main_model: str = ""               # 主模型名称（必须注册在 models 中）
    models: dict[str, ModelConfig] = field(default_factory=dict)
    deliberation: DeliberationConfig = field(default_factory=DeliberationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)

    def register_model(self, cfg: ModelConfig):
        self.models[cfg.name] = cfg


# ──────────────────────────────────────────────
# 内置预置配置
# ──────────────────────────────────────────────


def default_config() -> SymposiumConfig:
    """创建默认配置（适配用户环境）"""
    c = SymposiumConfig()

    # 本地主力推理（RTX 5090 - 27B）
    c.register_model(ModelConfig(
        name="local-27b",
        provider="local",
        endpoint="http://192.168.0.5:1234/v1",
        model_name="qwen3.6-27b-mtp",
        temperature=0.7,
        description="本地主力推理，快速响应，适合代码搜索和常规任务",
        weight=1.0,
    ))

    # 本地辅助推理（RTX 2070）
    c.register_model(ModelConfig(
        name="local-n2",
        provider="local",
        endpoint="http://127.0.0.1:12341/v1",
        model_name="qwopus3.6-27b-v2-mtp",
        temperature=0.7,
        description="本地辅助推理，适合并行探索和轻量任务",
        weight=0.8,
    ))

    # 远程高精度推理
    c.register_model(ModelConfig(
        name="remote-gpt5",
        provider="remote",
        endpoint="https://allincode.top/v1",
        model_name="gpt-5.4",
        temperature=0.7,
        description="远程高精度推理，长上下文，适合架构设计和复杂分析",
        weight=1.2,
    ))

    # 默认主模型
    c.main_model = "remote-gpt5"

    return c
