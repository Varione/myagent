"""
Debate Room — 受控讨论室。

Phase 3: 针对高不确定性或冲突问题，Supervisor 发起有限轮次的限定角色讨论，
最终由 Supervisor 裁决。

讨论规则：
- 由 Supervisor 发起
- 限定参与角色和讨论目标
- 限定轮数和输出格式
- 最终由 Supervisor 裁决并记录到 Decision Log
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .events import EventBus, WorkflowEvent


@dataclass
class DebateTurn:
    """单轮发言记录。"""

    round_number: int
    role: str
    model_id: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "role": self.role,
            "model_id": self.model_id,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class DebateResult:
    """讨论室裁决结果。"""

    debate_id: str
    topic: str
    status: str = "pending"  # pending | in_progress | resolved | timeout
    turns: list[DebateTurn] = field(default_factory=list)
    ruling: str = ""
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "status": self.status,
            "turns": [t.to_dict() for t in self.turns],
            "ruling": self.ruling,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


class DebateRoom:
    """
    受控讨论室：Supervisor 发起，限定角色、轮数和输出格式。

    Usage:
        room = DebateRoom(event_bus=event_bus)
        room.initiate(
            topic="Worker vs Critic 关于方案 A/B 的选择",
            participants=["Worker", "Critic"],
            max_rounds=2,
        )
        # 外部调用者依次提交各角色发言：
        room.add_turn(role="Worker", model_id="m1", content="我认为...")
        room.add_turn(role="Critic", model_id="m2", content="我反对...")
        # Supervisor 裁决：
        room.resolve(ruling="采用方案 A", rationale="...")
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self._active_debates: dict[str, DebateResult] = {}

    # ── 发起讨论 ─────────────────────────────────────────────────────

    def initiate(
        self,
        topic: str,
        participants: list[str],
        max_rounds: int = 2,
        debate_id: Optional[str] = None,
        context_entries: Optional[list[str]] = None,
    ) -> DebateResult:
        """
        Supervisor 发起一场受控讨论。

        Args:
            topic: 讨论主题/争议点
            participants: 参与角色列表（如 ["Worker", "Critic"]）
            max_rounds: 最大轮数（默认 2）
            debate_id: 可选，自定义 ID
            context_entries: 相关黑板条目引用

        Returns:
            DebateResult 实例
        """
        did = debate_id or f"DEB-{hash(topic) % 100000:05d}"
        result = DebateResult(
            debate_id=did,
            topic=topic,
            status="in_progress",
        )
        self._active_debates[did] = result

        # Publish event
        if self.event_bus:
            self.event_bus.publish(
                "debate_initiated",
                source="DebateRoom",
                payload={
                    "debate_id": did,
                    "topic": topic,
                    "participants": participants,
                    "max_rounds": max_rounds,
                    "context_entries": context_entries or [],
                },
            )

        return result

    # ── 添加发言 ─────────────────────────────────────────────────────

    def add_turn(
        self,
        debate_id: str,
        role: str,
        model_id: str,
        content: str,
    ) -> DebateTurn:
        """
        参与者提交一轮发言。

        Args:
            debate_id: 讨论 ID
            role: 角色名称
            model_id: 模型 ID
            content: 发言内容

        Returns:
            DebateTurn 实例
        """
        debate = self._active_debates.get(debate_id)
        if debate is None:
            raise ValueError(f"Debate '{debate_id}' not found")

        if debate.status != "in_progress":
            raise ValueError(f"Debate '{debate_id}' is not active (status={debate.status})")

        turn = DebateTurn(
            round_number=len(debate.turns) + 1,
            role=role,
            model_id=model_id,
            content=content,
        )
        debate.turns.append(turn)

        if self.event_bus:
            self.event_bus.publish(
                "debate_turn",
                source="DebateRoom",
                payload={
                    "debate_id": debate_id,
                    "turn_number": turn.round_number,
                    "role": role,
                    "model_id": model_id,
                },
            )

        return turn

    # ── 裁决 ─────────────────────────────────────────────────────────

    def resolve(
        self,
        debate_id: str,
        ruling: str,
        rationale: str = "",
        evidence_refs: Optional[list[str]] = None,
    ) -> DebateResult:
        """
        Supervisor 对讨论做出最终裁决。

        Args:
            debate_id: 讨论 ID
            ruling: 裁决结论
            rationale: 裁决理由
            evidence_refs: 证据引用列表（黑板条目、产物 ID 等）

        Returns:
            更新后的 DebateResult
        """
        debate = self._active_debates.get(debate_id)
        if debate is None:
            raise ValueError(f"Debate '{debate_id}' not found")

        debate.status = "resolved"
        debate.ruling = ruling
        debate.rationale = rationale
        debate.evidence_refs = evidence_refs or []
        debate.resolved_at = time.time()

        if self.event_bus:
            self.event_bus.publish(
                "debate_resolved",
                source="DebateRoom",
                payload={
                    "debate_id": debate_id,
                    "ruling": ruling,
                    "rationale": rationale,
                    "turn_count": len(debate.turns),
                },
            )

        return debate

    # ── 查询 ─────────────────────────────────────────────────────────

    def get_debate(self, debate_id: str) -> Optional[DebateResult]:
        """获取指定讨论。"""
        return self._active_debates.get(debate_id)

    def list_active_debates(self) -> list[DebateResult]:
        """列出所有活跃中的讨论。"""
        return [d for d in self._active_debates.values() if d.status == "in_progress"]

    def list_resolved_debates(self) -> list[DebateResult]:
        """列出已裁决的讨论。"""
        return [d for d in self._active_debates.values() if d.status == "resolved"]

    def all_debates(self) -> list[DebateResult]:
        """列出所有讨论。"""
        return list(self._active_debates.values())

    # ── 辅助方法 ─────────────────────────────────────────────────────

    def get_turns_by_role(self, debate_id: str, role: str) -> list[DebateTurn]:
        """获取某角色在某讨论中的所有发言。"""
        debate = self._active_debates.get(debate_id)
        if debate is None:
            return []
        return [t for t in debate.turns if t.role == role]

    def turn_count(self, debate_id: str) -> int:
        """返回某讨论的发言轮数。"""
        debate = self._active_debates.get(debate_id)
        return len(debate.turns) if debate else 0

    def has_reached_max_rounds(self, debate_id: str, max_rounds: int) -> bool:
        """检查是否已达到最大轮数。"""
        return self.turn_count(debate_id) >= max_rounds
