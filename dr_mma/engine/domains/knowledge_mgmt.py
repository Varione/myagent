"""
Knowledge Management Agent - 知识管理专业 Agent。

Phase 5: 领域专业化。支持文档摘要、实体提取、问答检索、知识图谱构建。

核心功能：
- 文档摘要：基于 TF 评分的关键句提取
- 实体提取：正则表达式驱动的名称/日期/数字/地点识别
- 问答检索：关键词重叠匹配
- 知识图谱：邻接表表示，BFS 路径搜索
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Optional

from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainTask,
    DomainType,
)


class KnowledgeGraph:
    """轻量级知识图谱（邻接表实现）。"""

    def __init__(self):
        self._nodes: set[str] = set()
        self._edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._reverse: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_node(self, name: str) -> None:
        self._nodes.add(name)

    def add_edge(self, source: str, target: str, label: str = "") -> None:
        self._nodes.add(source)
        self._nodes.add(target)
        self._edges[source].append((target, label))
        self._reverse[target].append((source, label))

    def get_neighbors(self, node: str) -> list[tuple[str, str]]:
        return self._edges.get(node, [])

    def get_incoming(self, node: str) -> list[tuple[str, str]]:
        return self._reverse.get(node, [])

    def has_node(self, node: str) -> bool:
        return node in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())

    def find_path(self, start: str, end: str) -> Optional[list[str]]:
        """BFS 最短路径搜索。"""
        if start not in self._nodes or end not in self._nodes:
            return None
        if start == end:
            return [start]

        visited = {start}
        queue = deque([[start]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor, _ in self._edges.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)

        return None

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self._nodes),
            "edges": [
                {"source": s, "target": t, "label": l}
                for s, targets in self._edges.items()
                for t, l in targets
            ],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }


class KnowledgeMgmtAgent(DomainAgent):
    """知识管理专业 Agent。"""

    def __init__(self, agent_id: str = "knowledge_mgmt_1"):
        super().__init__(agent_id, DomainType.KNOWLEDGE_MGMT)
        self.profile.skills = self.get_domain_skills()

    def get_domain_skills(self) -> dict[str, float]:
        return {
            "document_summarization": 0.85,
            "knowledge_graph": 0.7,
            "qa_system": 0.8,
            "entity_extraction": 0.75,
            "relation_mapping": 0.65,
        }

    def get_calibration_tasks(self) -> list[dict]:
        return [
            {
                "name": "calib_summarization",
                "objective": "验证文档摘要生成能力",
                "input": {
                    "subtask": "document_summarization",
                    "text": "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks with many layers. Natural language processing enables computers to understand text.",
                },
            },
            {
                "name": "calib_entity_extraction",
                "objective": "验证实体提取能力",
                "input": {
                    "subtask": "entity_extraction",
                    "text": "John Smith was born on 1990-05-15 in New York. He graduated with a GPA of 3.8 from MIT.",
                },
            },
            {
                "name": "calib_qa_retrieval",
                "objective": "验证问答检索能力",
                "input": {
                    "subtask": "qa_retrieval",
                    "knowledge_base": [
                        {"question": "What is Python?", "answer": "A programming language."},
                        {"question": "How to install pip?", "answer": "Use python -m ensurepip."},
                    ],
                    "query": "install pip package manager",
                },
            },
        ]

    def execute_domain_task(self, task: DomainTask) -> dict:
        subtask = task.input_data.get("subtask", "")

        if subtask == "document_summarization":
            return self._execute_summarization(task.input_data)
        elif subtask == "entity_extraction":
            return self._execute_entity_extraction(task.input_data)
        elif subtask == "qa_retrieval":
            return self._execute_qa_retrieval(task.input_data)
        elif subtask == "build_kg":
            return self._execute_build_kg(task.input_data)
        else:
            return {"error": "Unknown subtask: " + str(subtask)}

    # -- Document Summarization --

    def _execute_summarization(self, input_data: dict) -> dict:
        text = input_data.get("text", "")
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if not sentences:
            return {
                "summary": "",
                "key_points": [],
                "original_length": len(text),
                "summary_length": 0,
            }

        word_freq = self._compute_word_freq(sentences)
        sentence_scores = []
        for sent in sentences:
            words = re.findall(r'\b[a-z]+\b', sent.lower())
            if not words:
                sentence_scores.append(0.0)
                continue
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            sentence_scores.append(score)

        num_summaries = max(1, len(sentences) // 2)
        indexed = list(enumerate(sentence_scores))
        indexed.sort(key=lambda x: -x[1])
        top_indices = sorted([i for i, _ in indexed[:num_summaries]])

        key_points = [sentences[i] for i in top_indices]
        summary = ". ".join(key_points) + "."

        return {
            "summary": summary,
            "key_points": key_points,
            "original_length": len(text),
            "summary_length": len(summary),
        }

    def _compute_word_freq(self, sentences: list[str]) -> dict[str, float]:
        freq = defaultdict(int)
        for sent in sentences:
            for word in re.findall(r'\b[a-z]+\b', sent.lower()):
                if len(word) > 2:
                    freq[word] += 1
        total = sum(freq.values()) or 1
        return {w: c / total for w, c in freq.items()}

    # -- Entity Extraction --

    def _execute_entity_extraction(self, input_data: dict) -> dict:
        text = input_data.get("text", "")
        entities = []

        # PERSON: capitalized names (2+ words)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            entities.append({"text": m.group(), "type": "PERSON", "start": m.start(), "end": m.end()})

        # DATE: YYYY-MM-DD
        for m in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', text):
            entities.append({"text": m.group(), "type": "DATE", "start": m.start(), "end": m.end()})

        # NUMBER: digits with optional decimal
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            val = m.group()
            if len(val) <= 4 and not any(e["start"] == m.start() for e in entities):
                entities.append({"text": val, "type": "NUMBER", "start": m.start(), "end": m.end()})

        # LOCATION: known places
        location_patterns = [r'\b(New York|London|Paris|Beijing|Tokyo|Shanghai)\b']
        for pattern in location_patterns:
            for m in re.finditer(pattern, text):
                entities.append({"text": m.group(), "type": "LOCATION", "start": m.start(), "end": m.end()})

        entities.sort(key=lambda e: e["start"])

        count_by_type = defaultdict(int)
        for e in entities:
            count_by_type[e["type"]] += 1

        return {
            "entities": entities,
            "count_by_type": dict(count_by_type),
            "total_count": len(entities),
        }

    # -- QA Retrieval --

    def _execute_qa_retrieval(self, input_data: dict) -> dict:
        kb = input_data.get("knowledge_base", [])
        query = input_data.get("query", "")

        if not kb or not query:
            return {"answer": "", "confidence": 0.0, "source_index": -1}

        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        best_score = 0.0
        best_idx = 0

        for i, entry in enumerate(kb):
            q_words = set(re.findall(r'\b\w+\b', entry["question"].lower()))
            if not q_words:
                continue
            overlap = len(query_words & q_words)
            score = overlap / len(q_words)
            if score > best_score:
                best_score = score
                best_idx = i

        return {
            "answer": kb[best_idx]["answer"],
            "confidence": round(best_score, 4),
            "source_index": best_idx,
        }

    # -- Knowledge Graph Building --

    def _execute_build_kg(self, input_data: dict) -> dict:
        nodes = input_data.get("nodes", [])
        edges = input_data.get("edges", [])

        kg = KnowledgeGraph()
        for n in nodes:
            kg.add_node(n)
        for e in edges:
            source, target = e[0], e[1]
            label = e[2] if len(e) > 2 else ""
            kg.add_edge(source, target, label)

        return {
            "graph": kg.to_dict(),
            "node_count": kg.node_count,
            "edge_count": kg.edge_count,
        }

    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        issues = []

        if "error" in output:
            issues.append(output["error"])
            return False, issues

        subtask_hint = self._detect_subtask_from_output(output)

        if subtask_hint == "document_summarization":
            for key in ("summary", "key_points", "original_length", "summary_length"):
                if key not in output:
                    issues.append("Missing key: " + key)
            if "key_points" in output and not isinstance(output["key_points"], list):
                issues.append("key_points must be a list")

        elif subtask_hint == "entity_extraction":
            for key in ("entities", "count_by_type", "total_count"):
                if key not in output:
                    issues.append("Missing key: " + key)
            if "entities" in output:
                for e in output["entities"]:
                    if not all(k in e for k in ("text", "type", "start", "end")):
                        issues.append("Entity missing required fields")
                        break

        elif subtask_hint == "qa_retrieval":
            for key in ("answer", "confidence", "source_index"):
                if key not in output:
                    issues.append("Missing key: " + key)
            if "confidence" in output and not (0.0 <= output["confidence"] <= 1.0):
                issues.append("confidence must be between 0 and 1")

        elif subtask_hint == "build_kg":
            for key in ("graph", "node_count", "edge_count"):
                if key not in output:
                    issues.append("Missing key: " + key)

        else:
            if not output:
                issues.append("Empty output")

        return len(issues) == 0, issues

    def _detect_subtask_from_output(self, output: dict) -> str:
        if "summary" in output and "key_points" in output:
            return "document_summarization"
        if "entities" in output and "count_by_type" in output:
            return "entity_extraction"
        if "answer" in output and "confidence" in output:
            return "qa_retrieval"
        if "graph" in output and "node_count" in output:
            return "build_kg"
        return ""
