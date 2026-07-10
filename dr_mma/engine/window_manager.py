from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class WindowMessage:
    """上下文窗口中的一条消息。"""

    role: MessageRole
    content: str
    importance: float = 1.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self.estimate_tokens()

    def estimate_tokens(self) -> int:
        """粗略估算 token 数（4字符 ≈ 1 token）。"""
        return max(1, len(self.content) // 4)

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "importance": self.importance,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class WindowConfig:
    """上下文窗口配置。"""

    max_tokens: int = 128000
    reserve_tokens: int = 4096
    system_always_keep: bool = True
    min_keep_count: int = 5

    @property
    def usable_tokens(self) -> int:
        return self.max_tokens - self.reserve_tokens


@dataclass
class WindowSnapshot:
    """窗口快照，用于回放和调试。"""

    messages: list[dict]
    total_tokens: int
    dropped_count: int
    dropped_tokens: int
    strategy: str


class WindowManager:
    """上下文窗口管理器：维护消息列表，在超限时智能截断。"""

    def __init__(self, config: WindowConfig | None = None):
        self.config = config or WindowConfig()
        self._messages: list[WindowMessage] = []
        self._lock = threading.Lock()
        self._total_dropped = 0
        self._total_dropped_tokens = 0

    def add(
        self,
        role: MessageRole,
        content: str,
        importance: float = 1.0,
        metadata: dict | None = None,
    ) -> WindowMessage:
        """添加一条消息到窗口。"""
        msg = WindowMessage(
            role=role,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        with self._lock:
            self._messages.append(msg)
        self.trim_if_needed()
        return msg

    def add_system(self, content: str) -> WindowMessage:
        """添加系统消息（最高重要性，永不截断）。"""
        return self.add(MessageRole.SYSTEM, content, importance=5.0)

    def add_user(self, content: str, importance: float = 1.0) -> WindowMessage:
        return self.add(MessageRole.USER, content, importance=importance)

    def add_assistant(self, content: str, importance: float = 2.0) -> WindowMessage:
        return self.add(MessageRole.ASSISTANT, content, importance=importance)

    def add_tool(
        self, content: str, metadata: dict | None = None
    ) -> WindowMessage:
        return self.add(MessageRole.TOOL, content, importance=1.5, metadata=metadata)

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def usage_ratio(self) -> float:
        if self.config.max_tokens == 0:
            return 0.0
        return min(1.0, self.total_tokens / self.config.max_tokens)

    # -- 窗口压缩策略 --------------------------------------------------

    def trim_if_needed(self) -> Optional[WindowSnapshot]:
        """如果超出预算，自动执行混合压缩策略。"""
        if self.total_tokens <= self.config.usable_tokens:
            return None
        return self.trim_hybrid()

    def trim_by_importance(
        self, target_tokens: Optional[int] = None
    ) -> WindowSnapshot:
        """按重要性排序，丢弃低重要性消息直到回到预算内。"""
        target = target_tokens or self.config.usable_tokens
        with self._lock:
            protected = [
                m
                for m in self._messages
                if m.role == MessageRole.SYSTEM and self.config.system_always_keep
            ]
            candidates = [m for m in self._messages if m not in protected]
            candidates.sort(key=lambda m: m.importance)

            dropped: list[WindowMessage] = []
            kept_candidates: list[WindowMessage] = []
            current_tokens = sum(m.token_count for m in protected)

            # 先保留所有候选，再决定哪些要丢弃
            for msg in candidates:
                remaining_after_drop = len(protected) + len(candidates) - len(dropped) - 1
                if (
                    current_tokens + msg.token_count <= target
                    or remaining_after_drop <= self.config.min_keep_count
                ):
                    kept_candidates.append(msg)
                    current_tokens += msg.token_count
                else:
                    dropped.append(msg)

            self._messages = protected + kept_candidates
            dropped_tokens = sum(m.token_count for m in dropped)
            self._total_dropped += len(dropped)
            self._total_dropped_tokens += dropped_tokens

        return WindowSnapshot(
            messages=[m.to_dict() for m in self._messages],
            total_tokens=self.total_tokens,
            dropped_count=len(dropped),
            dropped_tokens=dropped_tokens,
            strategy="importance",
        )

    def trim_sliding(self, keep_tail: int = 20) -> WindowSnapshot:
        """滑动窗口：保留最近的 N 条消息 + 系统消息。"""
        with self._lock:
            protected = [
                m
                for m in self._messages
                if m.role == MessageRole.SYSTEM and self.config.system_always_keep
            ]
            non_system = [
                m
                for m in self._messages
                if m not in protected
            ]

            to_drop: list[WindowMessage] = []
            if len(non_system) > keep_tail:
                to_drop = non_system[:-keep_tail]

            dropped_tokens = sum(m.token_count for m in to_drop)
            self._messages = protected + non_system[-keep_tail:]
            self._total_dropped += len(to_drop)
            self._total_dropped_tokens += dropped_tokens

        return WindowSnapshot(
            messages=[m.to_dict() for m in self._messages],
            total_tokens=self.total_tokens,
            dropped_count=len(to_drop),
            dropped_tokens=dropped_tokens,
            strategy="sliding",
        )

    def trim_hybrid(self) -> WindowSnapshot:
        """混合策略：先按重要性丢弃，再滑动窗口兜底。"""
        snap1 = self.trim_by_importance()
        if self.total_tokens > self.config.usable_tokens:
            snap2 = self.trim_sliding(
                keep_tail=max(self.config.min_keep_count, 15)
            )
            snap1.dropped_count += snap2.dropped_count
            snap1.dropped_tokens += snap2.dropped_tokens
            snap1.strategy = "hybrid"
            snap1.messages = [m.to_dict() for m in self._messages]
            snap1.total_tokens = self.total_tokens
        else:
            snap1.strategy = "importance"
        return snap1

    # -- 查询与导出 ----------------------------------------------------

    def get_messages(
        self, role: Optional[MessageRole] = None
    ) -> list[WindowMessage]:
        if role is None:
            return list(self._messages)
        return [m for m in self._messages if m.role == role]

    def get_recent(self, n: int = 10) -> list[WindowMessage]:
        return list(self._messages[-n:])

    def to_prompt(self) -> str:
        parts = []
        for m in self._messages:
            prefix = f"[{m.role.value.upper()}]"
            parts.append(f"{prefix}\n{m.content}\n")
        return "".join(parts)

    def to_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def snapshot(self) -> WindowSnapshot:
        return WindowSnapshot(
            messages=[m.to_dict() for m in self._messages],
            total_tokens=self.total_tokens,
            dropped_count=0,
            dropped_tokens=0,
            strategy="none",
        )

    def clear(self) -> int:
        with self._lock:
            count = len(self._messages)
            self._messages = []
        return count

    @property
    def stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "message_count": self.message_count,
            "usage_ratio": round(self.usage_ratio, 4),
            "max_tokens": self.config.max_tokens,
            "usable_tokens": self.config.usable_tokens,
            "total_dropped": self._total_dropped,
            "total_dropped_tokens": self._total_dropped_tokens,
        }
