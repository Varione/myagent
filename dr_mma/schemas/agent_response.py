"""
AgentResponse Schema — 每个 agent 执行完子任务后的标准输出格式

固定字段定义，MVP 期间不允许修改。
"""

from dataclasses import dataclass, field, asdict
import json


VALID_STATUSES = {"completed", "failed", "need_review", "low_confidence", "skipped", "schema_error"}
VALID_ROLES = {"Planner", "Worker", "Critic", "Verifier", "Supervisor", "Researcher",
               "Domain Expert", ""}
VALID_SEVERITIES = {"low", "medium", "high"}


@dataclass
class Claim:
    """论断：单个观点或结论"""
    claim: str = ""
    confidence: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = []
        if not self.claim:
            errors.append("Claim.claim: 不能为空")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"Claim.confidence: 超出范围 [0, 1] ({self.confidence})")
        if not isinstance(self.evidence_refs, list):
            errors.append("Claim.evidence_refs: 必须是 list")
        return errors


@dataclass
class Risk:
    """风险提示"""
    risk: str = ""
    severity: str = "medium"     # low | medium | high
    mitigation: str = ""

    def validate(self) -> list[str]:
        errors = []
        if not self.risk:
            errors.append("Risk.risk: 不能为空")
        if self.severity not in VALID_SEVERITIES:
            errors.append(f"Risk.severity: 无效值 '{self.severity}'")
        return errors


@dataclass
class ToolCall:
    """工具调用请求"""
    tool_name: str = ""
    args: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = []
        if not self.tool_name:
            errors.append("ToolCall.tool_name: 不能为空")
        if not isinstance(self.args, dict):
            errors.append("ToolCall.args: 必须是 dict")
        return errors


@dataclass
class ArtifactRef:
    """产物引用"""
    artifact_id: str = ""
    version: int = 1

    def validate(self) -> list[str]:
        errors = []
        if not self.artifact_id:
            errors.append("ArtifactRef.artifact_id: 不能为空")
        if self.version < 0:
            errors.append(f"ArtifactRef.version: 不能为负 ({self.version})")
        return errors


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
        tool_calls: 工具调用请求列表
        tool_results: 工具执行结果列表（由工作流填充）
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
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResponse":
        claims = [Claim(**c) for c in data.get("claims", [])]
        artifacts = [ArtifactRef(**a) for a in data.get("artifacts", [])]
        risks = [Risk(**r) for r in data.get("risks", [])]
        tool_calls = [ToolCall(**t) for t in data.get("tool_calls", [])]
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
            tool_calls=tool_calls,
            tool_results=data.get("tool_results", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def is_success(self) -> bool:
        return self.status == "completed"

    def needs_review(self) -> bool:
        return self.status in ("need_review", "low_confidence")

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    # ── Schema Validation ────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the response and return a list of error messages (empty = valid)."""
        errors: list[str] = []

        if not self.task_id:
            errors.append("task_id: 不能为空")
        if self.role and self.role not in VALID_ROLES:
            errors.append(f"role: 无效角色 '{self.role}'")
        if self.status not in VALID_STATUSES:
            errors.append(f"status: 无效状态 '{self.status}'，有效值: {sorted(VALID_STATUSES)}")
        if len(self.summary) > 1000:
            errors.append(f"summary: 超过最大长度 ({len(self.summary)}/1000)")
        if len(self.content) > 500000:
            errors.append(f"content: 超过最大长度 ({len(self.content)}/500000)")

        # Validate nested objects
        for i, claim in enumerate(self.claims):
            for e in claim.validate():
                errors.append(f"claims[{i}]: {e}")
        for i, risk in enumerate(self.risks):
            for e in risk.validate():
                errors.append(f"risks[{i}]: {e}")
        for i, tc in enumerate(self.tool_calls):
            for e in tc.validate():
                errors.append(f"tool_calls[{i}]: {e}")
        for i, art in enumerate(self.artifacts):
            for e in art.validate():
                errors.append(f"artifacts[{i}]: {e}")

        if not isinstance(self.tool_results, list):
            errors.append("tool_results: 必须是 list")

        return errors

    def is_valid(self) -> bool:
        """Return True if the response passes validation."""
        return len(self.validate()) == 0
