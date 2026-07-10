"""
TaskContract Schema — 每个子任务的标准输入合同

固定字段定义，MVP 期间不允许修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import uuid


VALID_ROLES = {"Planner", "Worker", "Critic", "Verifier", "Supervisor", "Researcher", "Domain Expert"}
VALID_TASK_TYPES = {"general", "direct_answer", "planning", "execution", "review", "revision",
                    "verification", "research", "domain_review", "summary", "debate"}
VALID_STATUSES = {"pending", "running", "completed", "failed", "cancelled", "skipped"}


@dataclass
class TaskContract:
    """
    子任务合同：定义 agent 执行子任务时的输入边界和目标。

    Fields:
        task_id: 全局唯一任务 ID
        task_name: 子任务名称
        role: 分配的角色 (Planner | Worker | Critic | Verifier | Supervisor)
        objective: 子任务目标描述
        input_refs: 输入依赖的黑板条目 ID 列表
        success_criteria: 成功标准列表
        timeout_seconds: 最大执行时间
        review_required: 是否需要审查
    """
    task_id: str = ""
    task_name: str = ""
    task_type: str = "general"
    role: str = ""
    objective: str = ""
    input_refs: list[str] = field(default_factory=list)
    assigned_model: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    allowed_tools: Optional[list[str]] = None
    expected_output_schema: str = ""
    success_criteria: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    review_required: bool = True
    contract_version: str = "2.0"

    def __post_init__(self):
        if not self.task_id:
            from ..engine.id_utils import make_id
            self.task_id = make_id("T")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskContract":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def short_summary(self) -> str:
        return f"[{self.task_id}] {self.task_name} ({self.role})"

    # ── Schema Validation ────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the contract and return a list of error messages (empty = valid)."""
        errors: list[str] = []

        if not self.task_id:
            errors.append("task_id: 不能为空")
        if not self.task_name:
            errors.append("task_name: 不能为空")
        if not self.role:
            errors.append("role: 不能为空")
        elif self.role not in VALID_ROLES:
            errors.append(f"role: 无效角色 '{self.role}'，有效值: {sorted(VALID_ROLES)}")
        if not self.objective:
            errors.append("objective: 不能为空")
        else:
            obj_len = len(self.objective)
            if obj_len < 5:
                errors.append(f"objective: 太短 ({obj_len} chars)，至少需要 5 字符")
            if obj_len > 10000:
                errors.append(f"objective: 太长 ({obj_len} chars)，最多 10000 字符")
        if self.task_type not in VALID_TASK_TYPES:
            errors.append(f"task_type: 无效类型 '{self.task_type}'，有效值: {sorted(VALID_TASK_TYPES)}")
        if self.timeout_seconds < 1 or self.timeout_seconds > 3600:
            errors.append(f"timeout_seconds: 超出范围 ({self.timeout_seconds})，有效范围 1-3600")
        if not isinstance(self.input_refs, list):
            errors.append("input_refs: 必须是 list")
        if not isinstance(self.success_criteria, list):
            errors.append("success_criteria: 必须是 list")
        if self.allowed_tools is not None and not isinstance(self.allowed_tools, list):
            errors.append("allowed_tools: 必须是 list 或 None")
        if not isinstance(self.required_capabilities, list):
            errors.append("required_capabilities: 必须是 list")

        return errors

    def is_valid(self) -> bool:
        """Return True if the contract passes validation."""
        return len(self.validate()) == 0
