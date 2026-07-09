"""
TaskContract Schema — 每个子任务的标准输入合同

固定字段定义，MVP 期间不允许修改。
"""

from dataclasses import dataclass, field, asdict
import json
import uuid


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
    allowed_tools: list[str] = field(default_factory=list)
    expected_output_schema: str = ""
    success_criteria: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    review_required: bool = True
    contract_version: str = "1.0"

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"T-{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskContract":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def short_summary(self) -> str:
        return f"[{self.task_id}] {self.task_name} ({self.role})"
