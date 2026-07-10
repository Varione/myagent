"""
DecisionLog — 决策记录 (SQLite)

记录工作流执行过程中所有关键决策：任务拆分、冲突裁决、重试判定等。
使用 SQLite 存储，支持事务、索引查询和上下文管理器。每线程独立连接保证并发安全。
"""

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


# ── SQL ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS decision_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id   TEXT    NOT NULL UNIQUE,
        task_id     TEXT    NOT NULL,
        decision    TEXT    NOT NULL,
        rationale   TEXT    NOT NULL DEFAULT '',
        context     TEXT    NOT NULL DEFAULT '{}',
        timestamp   TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_decision_task_id   ON decision_log(task_id);
    CREATE INDEX IF NOT EXISTS idx_decision_timestamp  ON decision_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_decision_combined   ON decision_log(task_id, decision);
"""


# ── Data Object ─────────────────────────────────────────────────────────────

class DecisionRecord:
    """单条决策记录"""

    def __init__(self, task_id: str, decision: str, rationale: str,
                 context: dict = None, record_id: str = ""):
        self.record_id = record_id or ""
        self.task_id = task_id
        self.decision = decision
        self.rationale = rationale
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "decision": self.decision,
            "rationale": self.rationale,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        dr = cls(
            task_id=data.get("task_id", ""),
            decision=data.get("decision", ""),
            rationale=data.get("rationale", ""),
            context=data.get("context", {}),
            record_id=data.get("record_id", ""),
        )
        dr.timestamp = data.get("timestamp", dr.timestamp)
        return dr

    def __repr__(self) -> str:
        return f"<DecisionRecord {self.record_id} task={self.task_id} decision={self.decision}>"


# ── Store ───────────────────────────────────────────────────────────────────

class DecisionLog:
    """决策日志——记录所有关键决策 (SQLite 后端)"""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._is_open = True

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的独立连接（懒初始化）"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_CREATE_TABLE)
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    # -- Connection management --

    def close(self):
        """关闭当前线程的数据库连接"""
        if self._is_open and hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __del__(self):
        """析构时自动关闭连接"""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -- CRUD --

    def log(self, task_id: str, decision: str, rationale: str,
            context: dict = None) -> DecisionRecord:
        """记录一条决策"""
        record = DecisionRecord(task_id, decision, rationale, context)
        record.record_id = f"D-{uuid.uuid4().hex}"
        context_json = json.dumps(record.context, ensure_ascii=False)
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO decision_log (record_id, task_id, decision, rationale, context, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record.record_id, record.task_id, record.decision,
             record.rationale, context_json, record.timestamp),
        )
        conn.commit()
        return record

    def query(self, task_id: str = "", decision: str = "",
              limit: int = 0, offset: int = 0) -> list[DecisionRecord]:
        """查询决策记录

        Args:
            task_id:  按任务 ID 过滤
            decision: 按决策类型过滤
            limit:    返回条数上限 (0 = 全部)
            offset:   偏移量

        Returns:
            按时间升序排列的决策记录列表
        """
        conn = self._get_conn()
        conditions: list[str] = []
        params: list = []

        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)
        if decision:
            conditions.append("decision = ?")
            params.append(decision)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT record_id, task_id, decision, rationale, context, timestamp FROM decision_log{where} ORDER BY timestamp ASC"

        if limit > 0:
            query += f" LIMIT {limit} OFFSET {offset}"

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            record_dict = {
                "record_id": row[0],
                "task_id": row[1],
                "decision": row[2],
                "rationale": row[3],
                "context": json.loads(row[4]) if row[4] else {},
                "timestamp": row[5],
            }
            results.append(DecisionRecord.from_dict(record_dict))
        return results

    def get(self, record_id: str) -> Optional[DecisionRecord]:
        """按记录 ID 获取单条"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT record_id, task_id, decision, rationale, context, timestamp "
            "FROM decision_log WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return DecisionRecord.from_dict({
            "record_id": row[0], "task_id": row[1], "decision": row[2],
            "rationale": row[3], "context": json.loads(row[4]) if row[4] else {},
            "timestamp": row[5],
        })

    def count(self, task_id: str = "", decision: str = "") -> int:
        """统计记录条数"""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list = []
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)
        if decision:
            conditions.append("decision = ?")
            params.append(decision)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        row = conn.execute(
            f"SELECT COUNT(*) FROM decision_log{where}", params
        ).fetchone()
        return row[0] if row else 0

    def delete(self, record_id: str) -> bool:
        """删除一条记录"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM decision_log WHERE record_id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0

    def clear(self):
        """清空所有记录（谨慎使用）"""
        conn = self._get_conn()
        conn.execute("DELETE FROM decision_log")
        conn.commit()
