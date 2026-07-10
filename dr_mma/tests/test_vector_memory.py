"""Vector Memory Layer unit tests."""

import pytest
from dr_mma.engine.vector_memory import (
    MemoryEntry,
    TextVectorizer,
    VectorMemory,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_empty_vectors(self):
        assert cosine_similarity([], [1.0]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestTextVectorizer:
    def test_tokenize(self):
        v = TextVectorizer()
        tokens = v._tokenize("Hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_fit_transform(self):
        docs = [
            "machine learning is great",
            "deep learning is powerful",
            "natural language processing",
        ]
        v = TextVectorizer(max_features=20)
        vectors = v.fit_transform(docs)
        assert len(vectors) == 3
        assert len(vectors[0]) == len(vectors[1])

    def test_vector_normalization(self):
        docs = ["hello world", "world hello"]
        v = TextVectorizer()
        vecs = v.fit_transform(docs)
        # Vectors should be normalized
        import math
        norm = math.sqrt(sum(x * x for x in vecs[0]))
        assert abs(norm - 1.0) < 1e-6

    def test_empty_document(self):
        docs = ["hello world", ""]
        v = TextVectorizer()
        v.fit(docs)
        vec = v.transform("")
        assert all(x == 0 for x in vec)


class TestVectorMemory:
    def _setup_memory(self):
        mem = VectorMemory()
        mem.add("Python is a programming language", category="language")
        mem.add("Java is also a programming language", category="language")
        mem.add("The sky is blue today", category="observation")
        mem.add("Machine learning uses neural networks", category="ai")
        return mem

    def test_add_and_count(self):
        mem = VectorMemory()
        mem.add("test content")
        assert mem.entry_count == 1

    def test_add_with_category(self):
        mem = VectorMemory()
        mem.add("content", category="test_cat")
        e = mem.recent_entries(1)[0]
        assert e.category == "test_cat"

    def test_search_returns_results(self):
        mem = self._setup_memory()
        results = mem.search("programming language", top_k=2)
        assert len(results) > 0
        # Language entries should be most relevant
        assert any("language" in r[0].content.lower() for r in results)

    def test_search_category_filter(self):
        mem = self._setup_memory()
        results = mem.search("programming", category="language")
        assert all(r[0].category == "language" for r in results)

    def test_search_min_similarity(self):
        mem = self._setup_memory()
        results = mem.search("xyzzy nonsense", min_similarity=0.5)
        assert len(results) == 0

    def test_get_entry(self):
        mem = VectorMemory()
        e = mem.add("test")
        retrieved = mem.get_entry(e.entry_id)
        assert retrieved is not None
        assert retrieved.content == "test"

    def test_delete_entry(self):
        mem = VectorMemory()
        e = mem.add("to delete")
        assert mem.delete(e.entry_id) is True
        assert mem.get_entry(e.entry_id) is None

    def test_delete_nonexistent(self):
        mem = VectorMemory()
        assert mem.delete("nope") is False

    def test_clear(self):
        mem = self._setup_memory()
        mem.clear()
        assert mem.entry_count == 0

    def test_recent_entries(self):
        mem = VectorMemory()
        mem.add("first")
        mem.add("second")
        mem.add("third")
        recent = mem.recent_entries(2)
        assert len(recent) == 2

    def test_categories(self):
        mem = self._setup_memory()
        cats = mem.get_categories()
        assert "language" in cats
        assert cats["language"] == 2

    def test_summary(self):
        mem = self._setup_memory()
        s = mem.summary()
        assert s["total_entries"] == 4
        assert "categories" in s


class TestVectorMemoryEviction:
    def test_evict_oldest(self):
        mem = VectorMemory(max_entries=3)
        mem.add("first")
        mem.add("second")
        mem.add("third")
        mem.add("fourth")  # Should evict "first"
        assert mem.entry_count == 3


class TestVectorMemoryBatch:
    def test_add_many(self):
        mem = VectorMemory()
        entries = [
            {"content": "a"},
            {"content": "b"},
            {"content": "c"},
        ]
        results = mem.add_many(entries, category="batch")
        assert len(results) == 3
        assert mem.entry_count == 3


class TestMemoryEntry:
    def test_to_dict(self):
        e = MemoryEntry(entry_id="m1", content="test")
        d = e.to_dict()
        assert d["entry_id"] == "m1"
        assert d["content"] == "test"

    def test_importance_default(self):
        e = MemoryEntry(entry_id="m1", content="test")
        assert e.importance == 1.0
