"""
ArtifactStore — 产物线性版本管理

每个 artifact 有唯一 ID 和递增版本号。MVP 使用 JSONL 存储。
"""

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import json
import copy


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
            artifact_id=data["artifact_id"],
            version=data["version"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )
        v.created_at = data.get("created_at", v.created_at)
        return v


class ArtifactStore:
    """产物存储，维护线性版本链"""

    def __init__(self, filepath: str | Path):
        self._filepath = Path(filepath)
        self._artifacts: dict[str, list[ArtifactVersion]] = {}

        if self._filepath.exists():
            self._load()

    def _load(self):
        """从 JSONL 加载全部版本"""
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        av = ArtifactVersion.from_dict(data)
                        self._artifacts.setdefault(av.artifact_id, []).append(av)
        except (IOError, OSError):
            pass

    def _flush(self):
        """全量写回文件"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            for versions in self._artifacts.values():
                for av in versions:
                    f.write(json.dumps(av.to_dict(), ensure_ascii=False) + "\n")

    def save(self, artifact_id: str, content: str,
             metadata: dict = None) -> ArtifactVersion:
        """保存新版本（自动递增版本号）"""
        versions = self._artifacts.get(artifact_id, [])
        next_ver = (versions[-1].version + 1) if versions else 1
        av = ArtifactVersion(artifact_id, next_ver, content, metadata)
        self._artifacts.setdefault(artifact_id, []).append(av)
        self._flush()
        return av

    def get_latest(self, artifact_id: str) -> Optional[ArtifactVersion]:
        """获取最新版本"""
        versions = self._artifacts.get(artifact_id, [])
        return versions[-1] if versions else None

    def get_version(self, artifact_id: str, version: int) -> Optional[ArtifactVersion]:
        """获取指定版本"""
        versions = self._artifacts.get(artifact_id, [])
        for av in versions:
            if av.version == version:
                return av
        return None

    def list_versions(self, artifact_id: str) -> list[ArtifactVersion]:
        """列出所有版本（从旧到新）"""
        return list(self._artifacts.get(artifact_id, []))

    def list_artifacts(self) -> list[str]:
        return list(self._artifacts.keys())

    def count(self) -> int:
        return sum(len(v) for v in self._artifacts.values())
