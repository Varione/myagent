"""
DecisionLog — 决策记录

记录工作流执行过程中所有关键决策：任务拆分、冲突裁决、重试判定等。
MVP 使用 JSONL 文件存储。
"""

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import json


class DecisionRecord:
    """单条决策记录"""

    def __init__(self, task_id: str, decision: str, rationale: str,
                 context: dict = None):
        self.task_id = task_id
        self.decision = decision
        self.rationale = rationale
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "decision": self.decision,
            "rationale": self.rationale,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        dr = cls(
            task_id=data["task_id"],
            decision=data["decision"],
            rationale=data.get("rationale", ""),
            context=data.get("context", {}),
        )
        dr.timestamp = data.get("timestamp", dr.timestamp)
        return dr


class DecisionLog:
    """决策日志——记录所有关键决策"""

    def __init__(self, filepath: str | Path):
        self._filepath = Path(filepath)
        self._records: list[DecisionRecord] = []

        if self._filepath.exists():
            self._load()

    def _load(self):
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._records.append(DecisionRecord.from_dict(data))
        except (IOError, OSError):
            pass

    def _append_to_file(self, record: DecisionRecord):
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def log(self, task_id: str, decision: str, rationale: str,
            context: dict = None) -> DecisionRecord:
        """记录一条决策"""
        record = DecisionRecord(task_id, decision, rationale, context)
        self._records.append(record)
        self._append_to_file(record)
        return record

    def query(self, task_id: str = "", decision: str = "",
              limit: int = 0) -> list[DecisionRecord]:
        """查询决策记录"""
        results = list(self._records)
        if task_id:
            results = [r for r in results if r.task_id == task_id]
        if decision:
            results = [r for r in results if r.decision == decision]
        if limit > 0:
            results = results[-limit:]
        return results

    def count(self) -> int:
        return len(self._records)
