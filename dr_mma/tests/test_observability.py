"""Observability Layer unit tests."""

import pytest
from dr_mma.engine.observability import (
    Event,
    EventType,
    Node,
    Edge,
    DAGGraph,
    EventTracer,
    DiagnosticsPanel,
)
from dr_mma.engine.tools import ToolRegistry


class TestEvent:
    def test_event_to_dict(self):
        e = Event(event_type=EventType.TASK_STARTED, task_id="T1")
        d = e.to_dict()
        assert d["type"] == "task_started"
        assert d["task_id"] == "T1"


class TestDAGGraph:
    def test_add_node(self):
        dag = DAGGraph()
        dag.add_node("A", "Node A")
        assert dag.node_count == 1

    def test_update_node(self):
        dag = DAGGraph()
        dag.add_node("A", "Node A", status="pending")
        dag.update_node("A", status="completed")
        assert dag.get_node("A").status == "completed"

    def test_add_edge(self):
        dag = DAGGraph()
        dag.add_node("A", "A")
        dag.add_node("B", "B")
        dag.add_edge("A", "B")
        assert dag.edge_count == 1

    def test_dependencies(self):
        dag = DAGGraph()
        dag.add_node("A", "A")
        dag.add_node("B", "B")
        dag.add_node("C", "C")
        dag.add_edge("A", "B")
        dag.add_edge("C", "B")
        deps = dag.get_dependencies("B")
        assert set(deps) == {"A", "C"}

    def test_dependents(self):
        dag = DAGGraph()
        dag.add_node("A", "A")
        dag.add_node("B", "B")
        dag.add_edge("A", "B")
        deps = dag.get_dependents("A")
        assert deps == ["B"]

    def test_to_graph_data(self):
        dag = DAGGraph()
        dag.add_node("X", "X")
        data = dag.to_graph_data()
        assert len(data["nodes"]) == 1
        assert data["node_count"] == 1

    def test_critical_path_simple(self):
        dag = DAGGraph()
        dag.add_node("A", "A", duration_ms=100)
        dag.add_node("B", "B", duration_ms=200)
        dag.add_node("C", "C", duration_ms=50)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        path = dag.critical_path()
        assert "A" in path
        assert "B" in path
        assert "C" in path

    def test_critical_path_empty(self):
        dag = DAGGraph()
        assert dag.critical_path() == []


class TestEventTracer:
    def test_record_and_get(self):
        tracer = EventTracer()
        tracer.record(EventType.TASK_STARTED, task_id="T1")
        events = tracer.get_events(task_id="T1")
        assert len(events) == 1

    def test_filter_by_type(self):
        tracer = EventTracer()
        tracer.record(EventType.TASK_STARTED, task_id="T1")
        tracer.record(EventType.TASK_COMPLETED, task_id="T1")
        assert len(tracer.get_events(event_type=EventType.TASK_STARTED)) == 1

    def test_filter_by_agent(self):
        tracer = EventTracer()
        tracer.record(EventType.AGENT_ASSIGNED, agent_id="A1")
        tracer.record(EventType.AGENT_ASSIGNED, agent_id="A2")
        assert len(tracer.get_events(agent_id="A1")) == 1

    def test_timeline(self):
        tracer = EventTracer()
        tracer.record(EventType.TASK_STARTED, task_id="T1")
        tl = tracer.timeline(task_id="T1")
        assert len(tl) == 1
        assert isinstance(tl[0], dict)

    def test_max_events(self):
        tracer = EventTracer(max_events=5)
        for i in range(10):
            tracer.record(EventType.TASK_STARTED, task_id=f"T{i}")
        assert tracer.event_count == 5


class TestDiagnosticsPanel:
    def _setup_panel(self):
        tracer = EventTracer()
        dag = DAGGraph()
        dag.add_node("T1", "Research")
        dag.add_node("T2", "Design")
        dag.add_edge("T1", "T2")

        reg = ToolRegistry()
        reg.register("x", lambda args: 1)
        reg.call("x", task_id="T1")

        tracer.record(EventType.TASK_STARTED, task_id="T1")
        tracer.record(EventType.TASK_FAILED, task_id="T2")

        return DiagnosticsPanel(tracer=tracer, dag=dag, tool_registry=reg)

    def test_summary_contains_dag(self):
        panel = self._setup_panel()
        s = panel.get_summary()
        assert s["dag"]["nodes"] == 2

    def test_summary_contains_events(self):
        panel = self._setup_panel()
        s = panel.get_summary()
        assert s["events"]["total"] == 2

    def test_health_check_degraded(self):
        panel = self._setup_panel()
        h = panel.health_check()
        assert h["status"] == "degraded"
        assert len(h["issues"]) > 0

    def test_health_check_healthy(self):
        tracer = EventTracer()
        tracer.record(EventType.TASK_STARTED)
        panel = DiagnosticsPanel(tracer=tracer)
        h = panel.health_check()
        assert h["status"] == "healthy"

    def test_export_json(self):
        panel = self._setup_panel()
        import json
        data = json.loads(panel.export_json())
        assert "dag" in data
        assert "health" in data


class TestNodeEdge:
    def test_node_to_dict(self):
        n = Node(node_id="N1", label="Test")
        d = n.to_dict()
        assert d["id"] == "N1"

    def test_edge_to_dict(self):
        e = Edge(source="A", target="B")
        d = e.to_dict()
        assert d["source"] == "A"
