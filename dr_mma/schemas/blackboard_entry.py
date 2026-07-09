"""
BlackboardEntry Schema — 黑板条目标准格式

固定字段定义，MVP 期间不允许修改。
"""

from dataclasses import dataclass, field, asdict
import json
import uuid
from datetime import datetime, timezone


@dataclass
class BlackboardEntry:
    """
    黑板条目：存储 agent 执行过程中的所有中间结果。

    Fields:
        entry_id: 全局唯一条目 ID
        task_id: 关联的任务 ID
        source_role: 来源角色
        content_type: 内容类型 (task_output | critic_report | verification_report | decision)
        summary: 内容摘要
        payload: 完整数据（dict 格式）
        created_at: ISO 格式时间戳
    """
    entry_id: str = ""
    task_id: str = ""
    source_role: str = ""
    content_type: str = ""
    summary: str = ""
    payload: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = f"BB-{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BlackboardEntry":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_json_line(self) -> str:
        """JSONL 格式的一行"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
