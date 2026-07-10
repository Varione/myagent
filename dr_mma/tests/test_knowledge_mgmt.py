"""Knowledge Management Agent unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainRegistry,
    DomainTask,
    DomainType,
)
from dr_mma.engine.domains.knowledge_mgmt import (
    KnowledgeGraph,
    KnowledgeMgmtAgent,
)


class TestKnowledgeGraph:
    def test_add_node(self):
        kg = KnowledgeGraph()
        kg.add_node("A")
        assert kg.has_node("A")

    def test_add_edge(self):
        kg = KnowledgeGraph()
        kg.add_edge("A", "B", "related")
        assert kg.edge_count == 1
        assert kg.node_count >= 2

    def test_get_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_edge("A", "B", "x")
        kg.add_edge("A", "C", "y")
        neighbors = kg.get_neighbors("A")
        assert len(neighbors) == 2

    def test_find_path_exists(self):
        kg = KnowledgeGraph()
        kg.add_edge("A", "B")
        kg.add_edge("B", "C")
        path = kg.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_find_path_none(self):
        kg = KnowledgeGraph()
        kg.add_node("A")
        kg.add_node("D")
        assert kg.find_path("A", "D") is None

    def test_find_path_same_node(self):
        kg = KnowledgeGraph()
        kg.add_node("A")
        assert kg.find_path("A", "A") == ["A"]

    def test_to_dict(self):
        kg = KnowledgeGraph()
        kg.add_edge("X", "Y", "link")
        d = kg.to_dict()
        assert "nodes" in d
        assert "edges" in d


class TestKnowledgeMgmtAgent:
    def _agent(self):
        return KnowledgeMgmtAgent()

    def test_init(self):
        a = self._agent()
        assert a.domain == DomainType.KNOWLEDGE_MGMT

    def test_skills(self):
        a = self._agent()
        skills = a.get_domain_skills()
        assert "document_summarization" in skills
        assert len(skills) == 5

    def test_calibration_tasks(self):
        a = self._agent()
        tasks = a.get_calibration_tasks()
        assert len(tasks) >= 3


class TestSummarization:
    def _agent(self):
        return KnowledgeMgmtAgent()

    def test_basic_summary(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="summarize",
            objective="",
            input_data={
                "subtask": "document_summarization",
                "text": "Machine learning is great. Deep learning is powerful. NLP enables text understanding.",
            },
        ))
        assert "summary" in r
        assert len(r["key_points"]) > 0

    def test_empty_text(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={"subtask": "document_summarization", "text": ""},
        ))
        assert r["summary"] == ""
        assert r["key_points"] == []

    def test_validates_output(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "document_summarization",
                "text": "One sentence. Two sentences here. Three more words.",
            },
        ))
        passed, issues = a.validate_output(r)
        assert passed is True


class TestEntityExtraction:
    def _agent(self):
        return KnowledgeMgmtAgent()

    def test_person_extraction(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "entity_extraction",
                "text": "John Smith went to the store.",
            },
        ))
        persons = [e for e in r["entities"] if e["type"] == "PERSON"]
        assert len(persons) > 0

    def test_date_extraction(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "entity_extraction",
                "text": "The event was on 2025-01-15.",
            },
        ))
        dates = [e for e in r["entities"] if e["type"] == "DATE"]
        assert len(dates) > 0

    def test_location_extraction(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "entity_extraction",
                "text": "He lives in New York.",
            },
        ))
        locs = [e for e in r["entities"] if e["type"] == "LOCATION"]
        assert len(locs) > 0

    def test_count_by_type(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "entity_extraction",
                "text": "John Smith was born on 1990-05-15 in New York.",
            },
        ))
        assert "PERSON" in r["count_by_type"] or "DATE" in r["count_by_type"]

    def test_validates_output(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "entity_extraction",
                "text": "Alice went to Paris.",
            },
        ))
        passed, issues = a.validate_output(r)
        assert passed is True


class TestQARetrieval:
    def _agent(self):
        return KnowledgeMgmtAgent()

    def test_basic_retrieval(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "qa_retrieval",
                "knowledge_base": [
                    {"question": "What is Python?", "answer": "A programming language."},
                    {"question": "How to install pip?", "answer": "Use ensurepip."},
                ],
                "query": "What is Python?",
            },
        ))
        assert r["answer"] == "A programming language."
        assert r["confidence"] > 0

    def test_partial_match(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "qa_retrieval",
                "knowledge_base": [
                    {"question": "How to install pip?", "answer": "Use ensurepip."},
                ],
                "query": "install pip package",
            },
        ))
        assert r["source_index"] == 0
        assert r["confidence"] > 0

    def test_no_match(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "qa_retrieval",
                "knowledge_base": [
                    {"question": "What is AI?", "answer": "Artificial intelligence."},
                ],
                "query": "how to cook pasta",
            },
        ))
        assert r["confidence"] == 0.0

    def test_empty_kb(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "qa_retrieval",
                "knowledge_base": [],
                "query": "test",
            },
        ))
        assert r["source_index"] == -1

    def test_validates_output(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "qa_retrieval",
                "knowledge_base": [
                    {"question": "What is X?", "answer": "Something."},
                ],
                "query": "what is X",
            },
        ))
        passed, issues = a.validate_output(r)
        assert passed is True


class TestBuildKG:
    def _agent(self):
        return KnowledgeMgmtAgent()

    def test_build_graph(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "build_kg",
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B", "related"], ["B", "C", "linked"]],
            },
        ))
        assert r["node_count"] >= 3
        assert r["edge_count"] == 2

    def test_validates_output(self):
        a = self._agent()
        r = a.execute_domain_task(DomainTask(
            task_id="T1",
            domain_type=DomainType.KNOWLEDGE_MGMT,
            task_name="",
            objective="",
            input_data={
                "subtask": "build_kg",
                "nodes": ["X"],
                "edges": [],
            },
        ))
        passed, issues = a.validate_output(r)
        assert passed is True


class TestCalibration:
    def test_calibration_runs(self):
        a = KnowledgeMgmtAgent()
        results = a.calibrate()
        assert len(results) >= 3

    def test_calibration_updates_profile(self):
        a = KnowledgeMgmtAgent()
        a.calibrate()
        assert a.profile.sample_count > 0


class TestRegistryIntegration:
    def test_register_and_find(self):
        reg = DomainRegistry()
        a = KnowledgeMgmtAgent("km_1")
        reg.register(a)
        agents = reg.get_agents_by_domain(DomainType.KNOWLEDGE_MGMT)
        assert len(agents) == 1
