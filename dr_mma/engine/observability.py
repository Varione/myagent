"""
Observability Layer — 可观测层。

Phase 4: 事件追踪、DAG 可视化数据、诊断面板。

核心功能：
- 事件追踪：记录 DR-MMA 运行时的所有关键事件
- DAG 可视化：生成任务依赖图的可视化数据
- 诊断面板：提供运行时指标和状态汇总
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """事件类型。"""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_ASSIGNED = "agent_assigned"
    DEBATE_STARTED = "debate_started"
    DEBATE_ROUNDED = "debate_rounded"
    DEBATE_CONVERGED = "debate_converged"
    TOOL_CALLED = "tool_called"
    TOOL_FAILED = "tool_failed"
    BUDGET_CHECKED = "budget_checked"
    BUDGET_EXCEEDED = "budget_exceeded"
    PERMISSION_DENIED = "permission_denied"
    CONTEXT_SUMMARIZED = "context_summarized"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class Event:
    """运行时事件。"""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    task_id: str = ""
    agent_id: str = ""
    role: str = ""
    data: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "data": self.data,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
        }


@dataclass
class Node:
    """DAG 节点。"""

    node_id: str
    label: str
    role: str = ""
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "label": self.label,
            "role": self.role,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """DAG 边。"""

    source: str
    target: str
    label: str = ""
    edge_type: str = "dependency"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "type": self.edge_type,
        }


class DAGGraph:
    """
    DAG 图：追踪任务依赖关系。

    Usage:
        dag = DAGGraph()
        dag.add_node("T1", "Research Phase", role="Researcher")
        dag.add_node("T2", "Design Phase", role="Architect")
        dag.add_edge("T1", "T2", label="depends_on")

        data = dag.to_graph_data()  # for visualization
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, node_id: str, label: str, **kwargs) -> Node:
        node = Node(node_id=node_id, label=label, **kwargs)
        self._nodes[node_id] = node
        return node

    def update_node(self, node_id: str, **kwargs) -> Optional[Node]:
        if node_id in self._nodes:
            for k, v in kwargs.items():
                setattr(self._nodes[node_id], k, v)
            return self._nodes[node_id]
        return None

    def add_edge(
        self, source: str, target: str, label: str = "", edge_type: str = "dependency"
    ) -> Edge:
        edge = Edge(source=source, target=target, label=label, edge_type=edge_type)
        self._edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_dependencies(self, node_id: str) -> list[str]:
        """获取某节点的所有上游依赖。"""
        deps = []
        for e in self._edges:
            if e.target == node_id:
                deps.append(e.source)
        return deps

    def get_dependents(self, node_id: str) -> list[str]:
        """获取某节点的所有下游依赖。"""
        deps = []
        for e in self._edges:
            if e.source == node_id:
                deps.append(e.target)
        return deps

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def to_graph_data(self) -> dict:
        """返回可视化数据。"""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    def critical_path(self) -> list[str]:
        """计算关键路径（最长路径）。"""
        if not self._nodes:
            return []

        # 拓扑排序
        in_degree = {n: 0 for n in self._nodes}
        for e in self._edges:
            if e.target in in_degree:
                in_degree[e.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for e in self._edges:
                if e.source == node:
                    in_degree[e.target] -= 1
                    if in_degree[e.target] == 0:
                        queue.append(e.target)

        # 最长路径 DP
        dist = {n: self._nodes[n].duration_ms for n in order}
        prev = {n: None for n in order}

        for n in order:
            for e in self._edges:
                if e.source == n:
                    new_dist = dist[n] + self._nodes[e.target].duration_ms
                    if new_dist > dist[e.target]:
                        dist[e.target] = new_dist
                        prev[e.target] = n

        # 回溯最长路径
        end_node = max(dist, key=dist.get)
        path = []
        cur = end_node
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path


class EventTracer:
    """
    事件追踪器：记录 DR-MMA 运行时的所有关键事件。

    Usage:
        tracer = EventTracer()
        tracer.record(EventType.TASK_STARTED, task_id="T1", agent_id="A1")
        events = tracer.get_events(task_id="T1")
    """

    def __init__(self, max_events: int = 10000):
        self._events: list[Event] = []
        self._max_events = max_events

    def record(
        self,
        event_type: EventType,
        task_id: str = "",
        agent_id: str = "",
        role: str = "",
        data: Optional[dict] = None,
        duration_ms: float = 0.0,
        message: str = "",
    ) -> Event:
        event = Event(
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            role=role,
            data=data or {},
            duration_ms=duration_ms,
            message=message,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        return event

    def get_events(
        self,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> list[Event]:
        results = self._events
        if task_id:
            results = [e for e in results if e.task_id == task_id]
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results

    @property
    def event_count(self) -> int:
        return len(self._events)

    def timeline(self, task_id: Optional[str] = None) -> list[dict]:
        """返回时间线数据。"""
        events = self.get_events(task_id=task_id)
        return [e.to_dict() for e in events]


class DiagnosticsPanel:
    """
    诊断面板：提供运行时指标和状态汇总。

    Usage:
        panel = DiagnosticsPanel(tracer, dag, registry)
        summary = panel.get_summary()
    """

    def __init__(
        self,
        tracer: Optional[EventTracer] = None,
        dag: Optional[DAGGraph] = None,
        tool_registry=None,
    ):
        self.tracer = tracer
        self.dag = dag
        self.tool_registry = tool_registry

    def get_summary(self) -> dict:
        """返回完整诊断摘要。"""
        summary = {
            "timestamp": time.time(),
            "dag": {},
            "events": {},
            "tools": {},
        }

        if self.dag:
            summary["dag"] = {
                "nodes": self.dag.node_count,
                "edges": self.dag.edge_count,
                "critical_path": self.dag.critical_path(),
            }

        if self.tracer:
            events = self.tracer.get_events()
            by_type = {}
            for e in events:
                t = e.event_type.value
                by_type[t] = by_type.get(t, 0) + 1

            failed = sum(
                1
                for e in events
                if e.event_type in (
                    EventType.TASK_FAILED,
                    EventType.TOOL_FAILED,
                    EventType.VERIFICATION_FAILED,
                )
            )

            summary["events"] = {
                "total": len(events),
                "by_type": by_type,
                "failed_count": failed,
            }

        if self.tool_registry:
            summary["tools"] = self.tool_registry.usage_summary()

        return summary

    def health_check(self) -> dict:
        """健康检查。"""
        issues = []

        if self.tracer:
            failed_events = self.tracer.get_events(event_type=EventType.TASK_FAILED)
            if failed_events:
                issues.append(
                    f"{len(failed_events)} task(s) failed"
                )

            budget_exceeded = self.tracer.get_events(
                event_type=EventType.BUDGET_EXCEEDED
            )
            if budget_exceeded:
                issues.append(
                    f"{len(budget_exceeded)} budget exceeded event(s)"
                )

        status = "healthy" if not issues else "degraded"
        return {
            "status": status,
            "issues": issues,
            "timestamp": time.time(),
        }

    def export_json(self) -> str:
        """导出完整诊断数据为 JSON。"""
        data = self.get_summary()
        data["health"] = self.health_check()
        return json.dumps(data, indent=2, ensure_ascii=False)
