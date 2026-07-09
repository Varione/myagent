"""
Blackboard — 黑板持久化存储

MVP 使用 JSONL 文件作为存储后端，每条记录是一个 BlackboardEntry。
支持追加写入和按条件查询。
"""

from typing import Optional
from pathlib import Path
import json

from ..schemas.blackboard_entry import BlackboardEntry


class Blackboard:
    """黑板：agent 执行过程中的中间结果存储"""

    def __init__(self, filepath: str | Path):
        self._filepath = Path(filepath)
        self._entries: list[BlackboardEntry] = []

        # 如果文件存在，加载历史数据
        if self._filepath.exists():
            self._load()

    def _load(self):
        """从 JSONL 文件加载已有条目"""
        self._entries = []
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._entries.append(BlackboardEntry.from_dict(data))
                        except json.JSONDecodeError:
                            continue
        except (IOError, OSError):
            self._entries = []

    def _append_to_file(self, entry: BlackboardEntry):
        """追加一条记录到 JSONL 文件"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(entry.to_json_line() + "\n")

    # ── 写操作 ──

    def write(self, entry: BlackboardEntry) -> str:
        """写入一条记录，返回 entry_id"""
        self._entries.append(entry)
        self._append_to_file(entry)
        return entry.entry_id

    # ── 读操作 ──

    def read(self, entry_id: str) -> Optional[BlackboardEntry]:
        """按 entry_id 读取单条"""
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def query(
        self,
        task_id: str = "",
        source_role: str = "",
        content_type: str = "",
        limit: int = 0,
    ) -> list[BlackboardEntry]:
        """按条件查询黑板条目"""
        results = list(self._entries)

        if task_id:
            results = [e for e in results if e.task_id == task_id]
        if source_role:
            results = [e for e in results if e.source_role == source_role]
        if content_type:
            results = [e for e in results if e.content_type == content_type]
        if limit > 0:
            results = results[-limit:]

        return results

    def get_latest(self, task_id: str = "") -> Optional[BlackboardEntry]:
        """获取最新一条记录"""
        entries = self.query(task_id=task_id) if task_id else self._entries
        return entries[-1] if entries else None

    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        """清空所有条目（谨慎使用）"""
        self._entries = []
        if self._filepath.exists():
            self._filepath.unlink()
