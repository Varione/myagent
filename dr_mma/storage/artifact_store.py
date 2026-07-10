"""
ArtifactStore — 产物线性版本管理 (SQLite)

每个 artifact 有唯一 ID 和递增版本号。使用 SQLite 存储，保证版本链
完整无间隙（version 从 1 递增，每次 +1）。每线程独立连接保证并发安全。
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


# ── SQL ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS artifact (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        artifact_id  TEXT    NOT NULL,
        version      INTEGER NOT NULL,
        content      TEXT    NOT NULL,
        metadata     TEXT    NOT NULL DEFAULT '{}',
        created_at   TEXT    NOT NULL,
        UNIQUE(artifact_id, version)
    );
    CREATE INDEX IF NOT EXISTS idx_art_artifact_id ON artifact(artifact_id);
    CREATE INDEX IF NOT EXISTS idx_art_version     ON artifact(artifact_id, version);
    CREATE INDEX IF NOT EXISTS idx_art_created_at  ON artifact(created_at);
"""


# ── Data Object ─────────────────────────────────────────────────────────────

class ArtifactVersion:
    """单个产物版本"""

    def __init__(self, artifact_id: str, version: int, content: str,
                 metadata: dict = None):
        self.artifact_id = artifact_id
        self.version = version
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactVersion":
        v = cls(
            artifact_id=data.get("artifact_id", ""),
            version=data.get("version", 1),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )
        v.created_at = data.get("created_at", v.created_at)
        return v

    def __repr__(self) -> str:
        return f"<ArtifactVersion {self.artifact_id}#v{self.version}>"


# ── Store ───────────────────────────────────────────────────────────────────

class ArtifactStore:
    """产物存储，维护线性版本链 (SQLite 后端)"""

    def __init__(self, db_path: str | Path = None):
        self._db_path = Path(db_path) if db_path else Path(":memory:")
        self._is_open = False
        self._local = threading.local()
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # If given a directory, create a default database file inside it
            if self._db_path.is_dir():
                self._db_path = self._db_path / "artifact_store.db"

        conn = self._create_conn()
        self._local.conn = conn
        self._is_open = True

    def _create_conn(self) -> sqlite3.Connection:
        """创建一个新的数据库连接"""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_CREATE_TABLE)
        conn.commit()
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的独立连接（懒初始化）"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_conn()
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
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -- CRUD --

    def save(self, artifact_id: str, content: str,
             metadata: dict = None, version: int = None) -> ArtifactVersion:
        """保存新版本（自动递增版本号，保证版本链无缝）

        Args:
            artifact_id: 产物标识符
            content: 产物内容
            metadata: 附加元数据
            version: 显式指定版本号。为 None 时自动递增。
                     指定时必须等于当前最大版本 +1，否则抛出 ValueError。
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MAX(version) FROM artifact WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        max_ver = row[0] if row and row[0] is not None else 0

        if version is not None:
            expected = max_ver + 1
            if version != expected:
                raise ValueError(
                    f"版本不连续: artifact_id={artifact_id}, "
                    f"期望 v{expected}，收到 v{version}"
                )
            next_ver = version
        else:
            next_ver = max_ver + 1

        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        av = ArtifactVersion(artifact_id, next_ver, content, metadata or {})
        av.created_at = now

        conn.execute(
            "INSERT INTO artifact (artifact_id, version, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (artifact_id, next_ver, content, metadata_json, now),
        )
        conn.commit()
        return av

    def get_latest(self, artifact_id: str) -> Optional[ArtifactVersion]:
        """获取最新版本"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT artifact_id, version, content, metadata, created_at "
            "FROM artifact WHERE artifact_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (artifact_id,),
        ).fetchone()
        return self._row_to_version(row) if row else None

    def get_version(self, artifact_id: str, version: int) -> Optional[ArtifactVersion]:
        """获取指定版本"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT artifact_id, version, content, metadata, created_at "
            "FROM artifact WHERE artifact_id = ? AND version = ?",
            (artifact_id, version),
        ).fetchone()
        return self._row_to_version(row) if row else None

    def list_versions(self, artifact_id: str) -> list[ArtifactVersion]:
        """列出所有版本（从旧到新）"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT artifact_id, version, content, metadata, created_at "
            "FROM artifact WHERE artifact_id = ? "
            "ORDER BY version ASC",
            (artifact_id,),
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def list_artifacts(self) -> list[str]:
        """列出所有 artifact ID"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT artifact_id FROM artifact ORDER BY artifact_id"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        """统计所有版本总数"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()
        return row[0] if row else 0

    def delete_artifact(self, artifact_id: str) -> bool:
        """删除整个 artifact 及其所有版本"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM artifact WHERE artifact_id = ?", (artifact_id,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_version(self, artifact_id: str, version: int) -> bool:
        """删除指定版本"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM artifact WHERE artifact_id = ? AND version = ?",
            (artifact_id, version),
        )
        conn.commit()
        return cursor.rowcount > 0

    # -- Version Chain Integrity --

    def verify_chain(self, artifact_id: str) -> list[str]:
        """检查版本链完整性，返回间隙列表（空 = 完整）"""
        versions = self.list_versions(artifact_id)
        if not versions:
            return []
        issues: list[str] = []
        for i, v in enumerate(versions):
            expected = i + 1
            if v.version != expected:
                issues.append(
                    f"版本间隙: 期望 v{expected}，实际 v{v.version} "
                    f"(间隙在索引 {i})"
                )
        # Check for duplicates
        version_set = {v.version for v in versions}
        if len(version_set) != len(versions):
            issues.append("版本重复: 存在相同版本号的记录")
        return issues

    def chain_is_valid(self, artifact_id: str) -> bool:
        """返回版本链是否完整无间隙"""
        return len(self.verify_chain(artifact_id)) == 0

    def get_latest_version(self, artifact_id: str) -> int:
        """返回当前最大版本号，不存在时返回 0"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MAX(version) FROM artifact WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0

    # -- Helper --

    @staticmethod
    def _row_to_version(row: tuple) -> ArtifactVersion:
        return ArtifactVersion(
            artifact_id=row[0],
            version=row[1],
            content=row[2],
            metadata=json.loads(row[3]) if row[3] else {},
        )
