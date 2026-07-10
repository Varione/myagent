"""
Blackboard — 黑板持久化存储 (SQLite)

使用 SQLite 存储 agent 执行过程中的中间结果。支持事务、索引查询、
上下文管理器和按条件过滤。每线程独立连接保证并发安全。
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..schemas.blackboard_entry import BlackboardEntry


# ── SQL ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS blackboard (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id      TEXT    NOT NULL UNIQUE,
        task_id       TEXT    NOT NULL DEFAULT '',
        source_role   TEXT    NOT NULL DEFAULT '',
        content_type  TEXT    NOT NULL DEFAULT '',
        summary       TEXT    NOT NULL DEFAULT '',
        payload       TEXT    NOT NULL DEFAULT '{}',
        created_at    TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_bb_entry_id     ON blackboard(entry_id);
    CREATE INDEX IF NOT EXISTS idx_bb_task_id      ON blackboard(task_id);
    CREATE INDEX IF NOT EXISTS idx_bb_source_role  ON blackboard(source_role);
    CREATE INDEX IF NOT EXISTS idx_bb_content_type ON blackboard(content_type);
    CREATE INDEX IF NOT EXISTS idx_bb_created_at   ON blackboard(created_at);
"""


class Blackboard:
    """黑板：agent 执行过程中的中间结果存储 (SQLite 后端)"""

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

    # ── 写操作 ──

    def write(self, entry: BlackboardEntry) -> str:
        """写入一条记录，返回 entry_id"""
        conn = self._get_conn()
        payload_json = json.dumps(entry.payload, ensure_ascii=False)
        conn.execute(
            "INSERT INTO blackboard (entry_id, task_id, source_role, content_type, summary, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry.entry_id, entry.task_id, entry.source_role,
             entry.content_type, entry.summary, payload_json, entry.created_at),
        )
        conn.commit()
        return entry.entry_id

    # ── 读操作 ──

    def read(self, entry_id: str) -> Optional[BlackboardEntry]:
        """按 entry_id 读取单条"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT entry_id, task_id, source_role, content_type, summary, payload, created_at "
            "FROM blackboard WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def query(
        self,
        task_id: str = "",
        source_role: str = "",
        content_type: str = "",
        limit: int = 0,
        offset: int = 0,
    ) -> list[BlackboardEntry]:
        """按条件查询黑板条目"""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list = []

        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)
        if source_role:
            conditions.append("source_role = ?")
            params.append(source_role)
        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT entry_id, task_id, source_role, content_type, summary, payload, created_at FROM blackboard{where} ORDER BY created_at ASC"

        if limit > 0:
            query += f" LIMIT {limit} OFFSET {offset}"

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_latest(self, task_id: str = "") -> Optional[BlackboardEntry]:
        """获取最新一条记录"""
        conn = self._get_conn()
        query = "SELECT entry_id, task_id, source_role, content_type, summary, payload, created_at FROM blackboard"
        params: list = []
        if task_id:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY id DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return self._row_to_entry(row) if row else None

    # ── 统计与维护 ──

    def count(self) -> int:
        """统计条目数"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM blackboard").fetchone()
        return row[0] if row else 0

    def clear(self):
        """清空所有条目（SQLite 后端保留数据库文件，只清除数据）"""
        conn = self._get_conn()
        conn.execute("DELETE FROM blackboard")
        conn.commit()

    def delete(self, entry_id: str) -> bool:
        """删除指定条目"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM blackboard WHERE entry_id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ── Helper ──

    @staticmethod
    def _row_to_entry(row: tuple) -> BlackboardEntry:
        return BlackboardEntry(
            entry_id=row[0],
            task_id=row[1],
            source_role=row[2],
            content_type=row[3],
            summary=row[4],
            payload=json.loads(row[5]) if row[5] else {},
            created_at=row[6],
        )
