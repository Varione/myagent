"""
Context Epoch — 基线快照与增量协调机制。

Phase X: 管理对话上下文的 epoch 生命周期，支持基线渲染、快照比较、
中途更新注入以及 epoch 终止条件检测。

核心概念：
- Baseline: 每个 epoch 的系统级上下文基线（角色定义、任务目标等）
- Snapshot: 当前上下文的完整快照，用于与基线比较
- Mid-conversation update: 在 epoch 内注入增量上下文变更
- Termination: 检测 epoch 是否应终止（token 超限、错误累积、超时等）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BaselineContext:
    """系统级上下文基线。"""

    system_prompt: str = ""
    role_definitions: dict[str, str] = field(default_factory=dict)
    task_objective: str = ""
    constraints: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "role_definitions": self.role_definitions,
            "task_objective": self.task_objective,
            "constraints": self.constraints,
            "available_tools": self.available_tools,
            "metadata": self.metadata,
        }

    def render(self) -> str:
        """渲染基线为字符串，用于注入上下文。"""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        if self.role_definitions:
            role_text = "; ".join(
                f"{role}: {desc}" for role, desc in self.role_definitions.items()
            )
            parts.append(f"Roles: {role_text}")
        if self.task_objective:
            parts.append(f"Objective: {self.task_objective}")
        if self.constraints:
            parts.append(f"Constraints: {'; '.join(self.constraints)}")
        if self.available_tools:
            parts.append(f"Tools: {', '.join(self.available_tools)}")
        return "\n".join(parts)


@dataclass
class ContextSnapshot:
    """上下文的完整快照。"""

    epoch_id: str
    token_count: int = 0
    message_count: int = 0
    tool_call_count: int = 0
    error_count: int = 0
    content_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "token_count": self.token_count,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "error_count": self.error_count,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def diff(self, other: ContextSnapshot) -> dict[str, Any]:
        """与另一个快照比较，返回差异摘要。"""
        return {
            "token_delta": self.token_count - other.token_count,
            "message_delta": self.message_count - other.message_count,
            "tool_call_delta": self.tool_call_count - other.tool_call_count,
            "error_delta": self.error_count - other.error_count,
            "time_delta_s": round(self.timestamp - other.timestamp, 2),
        }


@dataclass
class MidConversationUpdate:
    """对话中途注入的增量上下文变更。"""

    update_id: str
    epoch_id: str
    update_type: str = "context_change"  # context_change | tool_update | role_change
    content: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "update_id": self.update_id,
            "epoch_id": self.epoch_id,
            "update_type": self.update_type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class TerminationCondition:
    """Epoch 终止条件检测器。"""

    def __init__(
        self,
        max_tokens: int = 128_000,
        max_errors: int = 5,
        max_duration_s: float = 3600.0,
        custom_check: Optional[Callable[[ContextSnapshot], bool]] = None,
    ):
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.max_duration_s = max_duration_s
        self.custom_check = custom_check

    def should_terminate(self, snapshot: ContextSnapshot, start_time: float) -> tuple[bool, str]:
        """检测是否应终止当前 epoch。

        Returns:
            (should_terminate, reason)
        """
        if snapshot.token_count >= self.max_tokens:
            return True, f"Token limit reached: {snapshot.token_count}/{self.max_tokens}"
        if snapshot.error_count >= self.max_errors:
            return True, f"Error threshold exceeded: {snapshot.error_count}/{self.max_errors}"
        elapsed = time.time() - start_time
        if elapsed >= self.max_duration_s:
            return True, f"Duration limit reached: {elapsed:.1f}s/{self.max_duration_s}s"
        if self.custom_check and self.custom_check(snapshot):
            return True, "Custom termination condition met"
        return False, ""


class ContextEpoch:
    """
    管理对话上下文的 epoch 生命周期。

    每个 epoch 包含：
    - baseline: 系统级上下文基线
    - rendered_baseline: 渲染后的基线字符串
    - snapshot: 当前上下文快照
    - terminated_at: epoch 终止时间（若已终止）

    Usage:
        epoch = ContextEpoch(
            epoch_id="E-001",
            baseline=BaselineContext(system_prompt="You are a helpful assistant"),
        )

        # 检查是否应终止
        should_stop, reason = epoch.should_terminate()

        # 注入中途更新
        epoch.apply_update(MidConversationUpdate(...))
    """

    def __init__(
        self,
        epoch_id: str,
        baseline: Optional[BaselineContext] = None,
        max_tokens: int = 128_000,
        max_errors: int = 5,
        max_duration_s: float = 3600.0,
        custom_check: Optional[Callable[[ContextSnapshot], bool]] = None,
    ):
        self.epoch_id = epoch_id
        self.baseline = baseline or BaselineContext()
        self.rendered_baseline = self.baseline.render()
        self.snapshot = ContextSnapshot(epoch_id=epoch_id)
        self.terminated_at: Optional[float] = None
        self.termination_reason: str = ""
        self.start_time = time.time()
        self._updates: list[MidConversationUpdate] = []
        self._termination = TerminationCondition(
            max_tokens=max_tokens,
            max_errors=max_errors,
            max_duration_s=max_duration_s,
            custom_check=custom_check,
        )

    @property
    def is_terminated(self) -> bool:
        return self.terminated_at is not None

    @property
    def duration_s(self) -> float:
        end = self.terminated_at or time.time()
        return round(end - self.start_time, 2)

    def should_terminate(self) -> tuple[bool, str]:
        """检测当前 epoch 是否应终止。"""
        if self.is_terminated:
            return True, self.termination_reason
        return self._termination.should_terminate(self.snapshot, self.start_time)

    def terminate(self, reason: str = "manual") -> None:
        """手动终止 epoch。"""
        if not self.is_terminated:
            self.terminated_at = time.time()
            self.termination_reason = reason

    def update_snapshot(
        self,
        token_count: Optional[int] = None,
        message_count: Optional[int] = None,
        tool_call_count: Optional[int] = None,
        error_count: Optional[int] = None,
        content_hash: Optional[str] = None,
    ) -> ContextSnapshot:
        """更新当前快照。"""
        if token_count is not None:
            self.snapshot.token_count = token_count
        if message_count is not None:
            self.snapshot.message_count = message_count
        if tool_call_count is not None:
            self.snapshot.tool_call_count = tool_call_count
        if error_count is not None:
            self.snapshot.error_count = error_count
        if content_hash is not None:
            self.snapshot.content_hash = content_hash
        return self.snapshot

    def apply_update(self, update: MidConversationUpdate) -> None:
        """注入中途上下文更新。"""
        if self.is_terminated:
            raise RuntimeError(f"Epoch {self.epoch_id} is already terminated")
        self._updates.append(update)

    def get_updates_since(self, since_timestamp: float) -> list[MidConversationUpdate]:
        """获取指定时间戳之后的所有更新。"""
        return [
            u for u in self._updates if u.timestamp > since_timestamp
        ]

    def get_all_updates(self) -> list[MidConversationUpdate]:
        """获取所有中途更新。"""
        return list(self._updates)

    def snapshot_diff(self, other: ContextEpoch) -> dict[str, Any]:
        """与另一个 epoch 的快照比较。"""
        return self.snapshot.diff(other.snapshot)

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "is_terminated": self.is_terminated,
            "terminated_at": self.terminated_at,
            "termination_reason": self.termination_reason,
            "duration_s": self.duration_s,
            "baseline": self.baseline.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "update_count": len(self._updates),
        }
