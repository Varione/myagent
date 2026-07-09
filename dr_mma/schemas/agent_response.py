"""
AgentResponse Schema — 每个 agent 执行完子任务后的标准输出格式

固定字段定义，MVP 期间不允许修改。
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class Claim:
    """论断：单个观点或结论"""
    claim: str = ""
    confidence: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class Risk:
    """风险提示"""
    risk: str = ""
    severity: str = "medium"     # low | medium | high
    mitigation: str = ""


@dataclass
class ArtifactRef:
    """产物引用"""
    artifact_id: str = ""
    version: int = 1


@dataclass
class AgentResponse:
    """
    Agent 输出协议：每个 agent 完成任务后必须按此格式输出。

    Fields:
        task_id: 对应的 TaskContract ID
        role: 当前角色
        status: completed | failed | need_review | low_confidence
        summary: 输出摘要
        content: 完整内容
        claims: 论断列表
        artifacts: 产物引用列表
        risks: 风险提示列表
        next_action_recommendation: 建议下一步操作
    """
    task_id: str = ""
    role: str = ""
    status: str = "completed"
    summary: str = ""
    content: str = ""
    claims: list[Claim] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)
    next_action_recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResponse":
        claims = [Claim(**c) for c in data.get("claims", [])]
        artifacts = [ArtifactRef(**a) for a in data.get("artifacts", [])]
        risks = [Risk(**r) for r in data.get("risks", [])]
        return cls(
            task_id=data.get("task_id", ""),
            role=data.get("role", ""),
            status=data.get("status", "completed"),
            summary=data.get("summary", ""),
            content=data.get("content", ""),
            claims=claims,
            artifacts=artifacts,
            risks=risks,
            next_action_recommendation=data.get("next_action_recommendation", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def is_success(self) -> bool:
        return self.status == "completed"

    def needs_review(self) -> bool:
        return self.status in ("need_review", "low_confidence")
