"""Vector Memory Layer — 向量知识库。

Phase 4: 历史记忆存储、相似检索、语义搜索。

核心功能：
- 向量存储：纯 Python 实现的轻量级向量索引
- 余弦相似度检索
- 记忆分类与过滤
- 上下文相关记忆召回
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryEntry:
    """记忆条目。"""

    entry_id: str
    content: str
    vector: list[float] = field(default_factory=list)
    category: str = "general"
    source_task: str = ""
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "vector_length": len(self.vector),
            "category": self.category,
            "source_task": self.source_task,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata,
        }


class TextVectorizer:
    """
    文本向量化器：使用 TF-IDF 风格的简单向量化。

    纯 Python 实现，无外部依赖。
    """

    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """简单分词。"""
        words = text.lower().split()
        return [w.strip(".,!?;:\"'()[]{}") for w in words if w.strip()]

    def fit(self, documents: list[str]) -> "TextVectorizer":
        """构建词汇表和 IDF。"""
        self.vocabulary = {}
        self.idf = {}
        self._doc_count = len(documents)

        df = {}
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        # 构建词汇表（按词频排序）
        sorted_tokens = sorted(df.items(), key=lambda x: -x[1])[: self.max_features]
        self.vocabulary = {t: i for i, (t, _) in enumerate(sorted_tokens)}

        # 计算 IDF
        for token, count in df.items():
            if token in self.vocabulary:
                self.idf[token] = math.log(
                    (1 + self._doc_count) / (1 + count)
                ) + 1

        return self

    def transform(self, text: str) -> list[float]:
        """将文本转换为向量。"""
        if not self.vocabulary:
            return []

        tokens = self._tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # 归一化 TF
        total = sum(tf.values()) or 1
        vector = [0.0] * len(self.vocabulary)
        for token, idx in self.vocabulary.items():
            if token in tf:
                vector[idx] = (tf[token] / total) * self.idf.get(token, 1.0)

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        vector = [v / norm for v in vector]
        return vector

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        """拟合并转换。"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1
    norm_b = math.sqrt(sum(y * y for y in b)) or 1
    return dot / (norm_a * norm_b)


class VectorMemory:
    """
    向量知识库：存储和检索记忆条目。

    Usage:
        memory = VectorMemory()
        memory.add("Python is a programming language", category="knowledge")
        results = memory.search("programming language", top_k=3)
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: dict[str, MemoryEntry] = {}
        self._vectorizer = TextVectorizer()
        self._max_entries = max_entries
        self._needs_reindex = False
        self._entry_counter = 0

    def add(
        self,
        content: str,
        category: str = "general",
        source_task: str = "",
        importance: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """添加记忆条目。"""
        self._entry_counter += 1
        entry_id = f"mem_{self._entry_counter:06d}"

        # 临时向量（基于当前词汇表）
        vector = self._vectorizer.transform(content) if self._vectorizer.vocabulary else []

        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            vector=vector,
            category=category,
            source_task=source_task,
            importance=importance,
            metadata=metadata or {},
        )

        self._entries[entry_id] = entry
        self._needs_reindex = True

        # 控制大小
        if len(self._entries) > self._max_entries:
            self._evict_oldest()

        return entry

    def add_many(
        self, entries: list[dict], category: str = "general"
    ) -> list[MemoryEntry]:
        """批量添加。"""
        results = []
        for e in entries:
            r = self.add(
                content=e["content"],
                category=category,
                source_task=e.get("source_task", ""),
                importance=e.get("importance", 1.0),
                metadata=e.get("metadata"),
            )
            results.append(r)
        return results

    def _reindex(self):
        """重建向量索引。"""
        if not self._needs_reindex or not self._entries:
            return

        docs = [e.content for e in self._entries.values()]
        vectors = self._vectorizer.fit_transform(docs)

        for entry, vector in zip(self._entries.values(), vectors):
            entry.vector = vector

        self._needs_reindex = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        语义搜索。

        Returns:
            (entry, similarity) 列表，按相似度降序排列。
        """
        self._reindex()

        if not self._entries or not self._vectorizer.vocabulary:
            return []

        query_vector = self._vectorizer.transform(query)
        if not query_vector:
            return []

        scored = []
        for entry in self._entries.values():
            if category and entry.category != category:
                continue

            sim = cosine_similarity(query_vector, entry.vector)
            # 重要性加权
            weighted_sim = sim * entry.importance

            if weighted_sim >= min_similarity:
                scored.append((entry, weighted_sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取指定条目。"""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """删除条目。"""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def clear(self):
        """清空所有记忆。"""
        self._entries.clear()
        self._vectorizer = TextVectorizer()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get_categories(self) -> dict[str, int]:
        """获取各类别条目数。"""
        cats = {}
        for e in self._entries.values():
            cats[e.category] = cats.get(e.category, 0) + 1
        return cats

    def _evict_oldest(self):
        """驱逐最旧的条目。"""
        if not self._entries:
            return

        oldest_id = min(
            self._entries, key=lambda k: self._entries[k].timestamp
        )
        del self._entries[oldest_id]

    def recent_entries(self, limit: int = 10) -> list[MemoryEntry]:
        """获取最近的条目。"""
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: -e.timestamp
        )
        return sorted_entries[:limit]

    def summary(self) -> dict:
        """返回记忆库摘要。"""
        return {
            "total_entries": self.entry_count,
            "categories": self.get_categories(),
            "vocabulary_size": len(self._vectorizer.vocabulary),
            "max_entries": self._max_entries,
        }
