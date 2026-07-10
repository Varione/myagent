"""
Compaction — 上下文压缩（滑动窗口 + 摘要生成）。

Phase X: 防止对话上下文膨胀，通过滑动窗口保留最近 N tokens，
对超出窗口的内容进行摘要压缩。

核心概念：
- SlidingWindow: 滑动窗口策略，保留最近 N tokens，丢弃最早的内容
- CompactionSummary: 压缩摘要，由模板生成
- CompactionTrigger: 触发策略（token 超限 / 手动触发）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TriggerMode(Enum):
    """Compaction 触发模式。"""

    AUTO = "auto"  # Token 超过阈值时自动触发
    MANUAL = "manual"  # 手动触发


@dataclass
class CompactionSummary:
    """压缩摘要结果。"""

    epoch_id: str
    summary_text: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    compressed_segments: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return round(self.tokens_after / self.tokens_before, 4)

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "summary_text": self.summary_text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "compressed_segments_count": len(self.compressed_segments),
            "timestamp": self.timestamp,
        }


@dataclass
class SlidingWindowConfig:
    """滑动窗口配置。"""

    max_tokens: int = 64_000
    keep_last_n_messages: int = 20
    summary_max_length: int = 500

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "keep_last_n_messages": self.keep_last_n_messages,
            "summary_max_length": self.summary_max_length,
        }


class CompactionTrigger:
    """
    Compaction 触发策略。

    支持两种模式：
    - AUTO: 当 token 数超过阈值时自动触发
    - MANUAL: 需显式调用 trigger() 方法
    """

    def __init__(
        self,
        mode: TriggerMode = TriggerMode.AUTO,
        token_threshold: int = 50_000,
    ):
        self.mode = mode
        self.token_threshold = token_threshold

    def should_compact(self, current_tokens: int) -> bool:
        if self.mode == TriggerMode.MANUAL:
            return False
        return current_tokens >= self.token_threshold

    def trigger(self) -> bool:
        """手动触发 compaction。"""
        return True


class CompactionEngine:
    """
    上下文压缩引擎。

    Usage:
        engine = CompactionEngine(
            config=SlidingWindowConfig(max_tokens=64000),
            trigger=CompactionTrigger(mode=TriggerMode.AUTO, token_threshold=50000),
        )

        # 添加消息到上下文
        engine.add_message("user", "Hello")
        engine.add_message("assistant", "Hi there!")

        # 检查是否需要压缩
        if engine.should_compact():
            summary = engine.compact()

    滑动窗口策略：
    - 保留最近 keep_last_n_messages 条消息
    - 对超出窗口的消息生成摘要并替换
    - 系统消息始终保留（不纳入窗口裁剪）
    """

    def __init__(
        self,
        config: Optional[SlidingWindowConfig] = None,
        trigger: Optional[CompactionTrigger] = None,
        summary_template: Optional[str] = None,
        custom_summary_fn: Optional[Callable[[list[dict]], str]] = None,
    ):
        self.config = config or SlidingWindowConfig()
        self.trigger = trigger or CompactionTrigger()
        self.summary_template = summary_template or (
            "[Compressed context] {count} earlier messages summarized: {summary}"
        )
        self.custom_summary_fn = custom_summary_fn

        # 消息存储：按角色分组，系统消息始终保留
        self._system_messages: list[str] = []
        self._messages: list[dict[str, Any]] = []
        self._compaction_history: list[CompactionSummary] = []

    @property
    def current_tokens(self) -> int:
        """估算当前上下文总 token 数。"""
        total_chars = sum(len(msg.get("content", "")) for msg in self._messages)
        total_chars += sum(len(s) for s in self._system_messages)
        return total_chars // 4

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def system_message_count(self) -> int:
        return len(self._system_messages)

    def add_system_message(self, content: str) -> None:
        """添加系统消息（始终保留，不参与窗口裁剪）。"""
        self._system_messages.append(content)

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """添加一条对话消息。"""
        msg: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._messages.append(msg)

    def should_compact(self) -> bool:
        """检查当前是否需要触发压缩。"""
        return self.trigger.should_compact(self.current_tokens)

    def compact(self, epoch_id: str = "default") -> CompactionSummary:
        """
        执行上下文压缩。

        策略：
        1. 保留系统消息（不裁剪）
        2. 保留最近 keep_last_n_messages 条消息
        3. 对超出窗口的消息生成摘要，插入一条压缩标记消息
        """
        tokens_before = self.current_tokens
        messages_to_compress = []

        if len(self._messages) > self.config.keep_last_n_messages:
            messages_to_compress = self._messages[: -self.config.keep_last_n_messages]
            # 保留最近的 N 条消息
            self._messages = self._messages[-self.config.keep_last_n_messages:]

        summary_text = ""
        if messages_to_compress:
            summary_text = self._generate_summary(messages_to_compress)
            # 插入压缩标记消息
            self.add_message(
                "system",
                f"[COMPACTED] {summary_text}",
                metadata={"compaction_marker": True},
            )

        tokens_after = self.current_tokens
        saved = max(0, tokens_before - tokens_after)

        summary = CompactionSummary(
            epoch_id=epoch_id,
            summary_text=summary_text,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compressed_segments=[m.get("role", "unknown") for m in messages_to_compress],
        )
        self._compaction_history.append(summary)
        return summary

    def compact_manual(self, epoch_id: str = "default") -> Optional[CompactionSummary]:
        """手动触发压缩（即使未达阈值）。"""
        if self.trigger.mode != TriggerMode.MANUAL and not self.should_compact():
            return None
        return self.compact(epoch_id)

    def get_messages(self, include_system: bool = True) -> list[dict[str, Any]]:
        """获取当前上下文消息列表。"""
        result = []
        if include_system:
            for s in self._system_messages:
                result.append({"role": "system", "content": s})
        result.extend(self._messages)
        return result

    def get_compaction_history(self) -> list[CompactionSummary]:
        """获取压缩历史。"""
        return list(self._compaction_history)

    def get_total_saved_tokens(self) -> int:
        """获取累计节省的 token 数。"""
        return sum(s.tokens_saved for s in self._compaction_history)

    def reset(self) -> None:
        """重置引擎状态。"""
        self._system_messages.clear()
        self._messages.clear()
        self._compaction_history.clear()

    def to_dict(self) -> dict:
        return {
            "current_tokens": self.current_tokens,
            "message_count": self.message_count,
            "system_message_count": self.system_message_count,
            "compaction_count": len(self._compaction_history),
            "total_saved_tokens": self.get_total_saved_tokens(),
            "config": self.config.to_dict(),
        }

    def _generate_summary(self, messages: list[dict[str, Any]]) -> str:
        """生成压缩摘要。"""
        if self.custom_summary_fn:
            return self.custom_summary_fn(messages)

        # 默认摘要策略：提取每条消息的前 N 个字符，拼接成摘要
        segments = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:80]
            segments.append(f"{role}: {content}")

        raw_summary = "; ".join(segments)
        if len(raw_summary) > self.config.summary_max_length:
            raw_summary = raw_summary[: self.config.summary_max_length] + "..."

        return self.summary_template.format(
            count=len(messages),
            summary=raw_summary,
        )
