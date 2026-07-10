"""Workflow engine with complexity routing, dynamic roles, and event logging."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import time
from typing import Optional

from ..models.adapter import ChatMessage, ModelAdapter
import threading
from ..roles.runner import RoleRunner
from ..schemas.agent_response import AgentResponse
from ..schemas.blackboard_entry import BlackboardEntry
from ..schemas.task_contract import TaskContract
from ..storage.artifact_store import ArtifactStore
from ..storage.blackboard import Blackboard
from ..storage.decision_log import DecisionLog
from .capabilities import CapabilityRegistry, DynamicRoleAssigner
from .complexity import (
    MODE_DIRECT,
    MODE_EXPANDED,
    MODE_SINGLE_REVIEW,
    MODE_STANDARD,
    ComplexityReport,
    TaskComplexityEvaluator,
)
from .events import (
    EVENT_LOW_CONFIDENCE,
    EVENT_NEED_REPLAN,
    EVENT_REVIEW_FAILED,
    EVENT_ROLE_ASSIGNED,
    EVENT_WORKFLOW_MODE,
    EventBus,
)
from .streaming import StreamSession
from .model_pool import ModelPool
from .permissions import PermissionManager
from .role_manager import RoleBinding, RoleMergerSplitter
from .tool_executor import ToolExecutor
from .tools import ToolRegistry, create_default_tool_registry
from .budget_controller import BudgetController
from .id_utils import make_id
from .context_manager import ContextManager
from .observability import EventTracer, DAGGraph, DiagnosticsPanel, EventType as ObsEventType
from .supervisor_modules import SupervisorOrchestrator


@dataclass
class WorkflowResult:
    """Result object returned by the workflow engine."""

    task_id: str = ""
    subtask_results: dict = field(default_factory=dict)
    final_output: str = ""
    blackboard_count: int = 0
    total_latency_ms: float = 0.0
    status: str = "completed"
    mode: str = ""
    complexity_score: int = 0
    role_assignments: dict[str, str] = field(default_factory=dict)
    event_count: int = 0
    dag_nodes: list[dict] = field(default_factory=list)
    runtime_config: dict = field(default_factory=dict)


class WorkflowEngine:
    """DR-MMA workflow engine."""

    def __init__(
        self,
        adapter: ModelAdapter,
        blackboard: Blackboard,
        artifact_store: ArtifactStore,
        decision_log: DecisionLog,
        main_model: str = "",
        runtime_config: Optional[dict] = None,
        model_pool: Optional[ModelPool] = None,
    ):
        self.adapter = adapter
        self.blackboard = blackboard
        self.artifact_store = artifact_store
        self.decision_log = decision_log
        self.main_model = main_model
        self.runner = RoleRunner(adapter)
        self.complexity_evaluator = TaskComplexityEvaluator()
        self.capability_registry = CapabilityRegistry()
        self.role_assigner = DynamicRoleAssigner(self.capability_registry)
        self.event_bus = EventBus()
        self.runtime_config = runtime_config or {}

        # Phase 2: Model Pool & Role Manager
        self.model_pool = model_pool or ModelPool()
        self.role_manager = RoleMergerSplitter(
            pool=self.model_pool,
            registry=self.capability_registry,
        )

        # Tool execution: bridge ToolRegistry + PermissionManager
        self.tool_registry = create_default_tool_registry()
        permission_mode = (runtime_config or {}).get("permission_mode", "workspace_only")
        self.permission_manager = PermissionManager(mode=permission_mode)
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            permission_manager=self.permission_manager,
        )

        # Phase 3: Budget, Context, Observability, Supervisor modules
        self.budget_controller = BudgetController(
            warning_threshold=float((runtime_config or {}).get("budget_warning_threshold", 0.8)),
        )
        self.context_manager = ContextManager(
            max_runtime_tokens=int((runtime_config or {}).get("max_runtime_tokens", 40_000)),
        )
        self.event_tracer = EventTracer()
        self.dag_graph = DAGGraph()
        self.diagnostics_panel = DiagnosticsPanel(
            tracer=self.event_tracer,
            dag=self.dag_graph,
            tool_registry=self.tool_registry,
        )
        self.supervisor = SupervisorOrchestrator()

        # Streaming session (set during execute())
        self.stream_session: Optional[StreamSession] = None
        self._cancel_token: Optional[threading.Event] = None
        self._current_workflow_id: str = ""

    def close(self):
        """Close all storage backends (SQLite connections)."""
        try:
            self.blackboard.close()
        except Exception:
            pass
        try:
            self.artifact_store.close()
        except Exception:
            pass
        try:
            self.decision_log.close()
        except Exception:
            pass

    def execute(
        self,
        user_task: str,
        primary_model: str = "",
        max_retries: int = 3,
        stream_session: "StreamSession" = None,
        cancel_token: threading.Event = None,
    ) -> WorkflowResult:
        import time

        t0 = time.time()
        primary_model = primary_model or self.main_model
        if not primary_model:
            raise ValueError("必须指定模型名称")

        # Phase 2: ensure all models are registered in the pool and health-checked
        self._ensure_pool_populated(primary_model)
        self.model_pool.health_check_all()

        result = WorkflowResult(task_id=make_id("wf"))
        result.runtime_config = dict(self.runtime_config)
        self._current_workflow_id = result.task_id

        # Use provided stream session or create one
        self.stream_session = stream_session or StreamSession(stream_id=result.task_id)
        self._cancel_token = cancel_token

        # Phase 3: Initialize Budget, Context, Tracing
        self.budget_controller.initialize(
            task_id=result.task_id,
            max_model_calls=int(self.runtime_config.get("max_model_calls", 30)),
            max_tool_calls=int(self.runtime_config.get("max_tool_calls", 10)),
            max_total_tokens=int(self.runtime_config.get("max_total_tokens", 120_000)),
            max_retries_per_node=max_retries + 1,
        )
        self.context_manager.init_global_context(result.task_id)
        self.event_tracer.record(
            ObsEventType.TASK_CREATED,
            task_id=result.task_id,
            message=f"Workflow created: mode pending, complexity pending",
        )

        complexity = self.complexity_evaluator.evaluate(user_task)
        assignments = self._assign_roles(primary_model, complexity)
        result.mode = complexity.mode
        result.complexity_score = complexity.score
        result.role_assignments = assignments

        # Phase 3: Supervisor task understanding
        understanding = self.supervisor.understanding.analyze(
            task_id=result.task_id,
            user_task=user_task,
        )
        self.context_manager.add_key_risk(result.task_id, f"risk_level={understanding.risk_level}")

        self.event_bus.publish(
            EVENT_WORKFLOW_MODE,
            source="ComplexityEvaluator",
            task_id=result.task_id,
            payload={"score": complexity.score, "mode": complexity.mode, "rationale": complexity.rationale},
        )
        self.decision_log.log(
            task_id=result.task_id,
            decision="mode_selected",
            rationale=f"{complexity.mode} (score={complexity.score})",
            context={"complexity": complexity.__dict__, "runtime": self.runtime_config},
        )

        if complexity.mode == MODE_DIRECT:
            return self._execute_direct(result, user_task, primary_model, complexity, t0)

        subtasks = self._plan_subtasks(result.task_id, user_task, assignments["Planner"], complexity)
        if not subtasks:
            # Phase 3: Record failure in observability
            self.event_tracer.record(
                ObsEventType.TASK_FAILED,
                task_id=result.task_id,
                data={"mode": complexity.mode},
                message="规划阶段未能生成有效子任务",
            )
            result.status = "failed"
            result.final_output = "规划阶段未能生成有效子任务。"
            result.total_latency_ms = round((time.time() - t0) * 1000, 1)
            return result

        # Phase 3: Supervisor DAG planning
        dag_plan = self.supervisor.planning.plan(result.task_id, understanding, subtasks)
        if dag_plan.nodes:
            self.decision_log.log(
                task_id=result.task_id,
                decision="dag_planned",
                rationale=f"DAG: {len(dag_plan.nodes)} nodes, {len(dag_plan.edges)} edges",
                context={"plan": dag_plan.to_dict()},
            )

        result.dag_nodes = [
            {
                "task_id": f"{result.task_id}-T{idx:02d}",
                "task_name": subtask.get("task_name", f"Task {idx + 1}"),
                "objective": subtask.get("objective", ""),
                "depends_on": list(subtask.get("depends_on", [])),
                "status": "pending",
            }
            for idx, subtask in enumerate(subtasks)
        ]

        if complexity.mode == MODE_SINGLE_REVIEW:
            subtasks = subtasks[:1]
        elif complexity.mode == MODE_STANDARD:
            subtasks = subtasks[: min(len(subtasks), 4)]
        elif complexity.mode == MODE_EXPANDED:
            subtasks = subtasks[: min(len(subtasks), 5)]
        result.dag_nodes = result.dag_nodes[: len(subtasks)]

        # Phase 3: Build DAG graph for observability
        self.dag_graph = DAGGraph()
        dep_map: dict[str, list[str]] = {}
        for node in result.dag_nodes:
            nid = node["task_id"]
            self.dag_graph.add_node(nid, node["task_name"], role="Worker", status="pending")
            dep_map[nid] = [d for d in node.get("depends_on", [])]
        # Infer edges from depends_on task names → task_ids
        id_by_name: dict[str, str] = {
            node["task_name"]: node["task_id"] for node in result.dag_nodes
        }
        for nid, deps in dep_map.items():
            for dep in deps:
                target_id = id_by_name.get(dep) if isinstance(dep, str) else dep
                if target_id and target_id in [n["task_id"] for n in result.dag_nodes]:
                    self.dag_graph.add_edge(target_id, nid, label="depends_on")

        self.event_tracer.record(
            ObsEventType.TASK_STARTED,
            task_id=result.task_id,
            data={"mode": complexity.mode, "subtask_count": len(subtasks)},
            message=f"Workflow started: {complexity.mode} with {len(subtasks)} subtasks",
        )

        subtask_results = self._execute_subtasks_dag(
            result=result,
            subtasks=subtasks,
            assignments=assignments,
            mode=complexity.mode,
            max_retries=max_retries,
            t0=t0,
        )

        # Check if DAG was cancelled
        if result.status == "cancelled":
            return result

        final_output = self._summarize(result.task_id, subtask_results, assignments["Supervisor"])
        result.subtask_results = subtask_results
        result.final_output = final_output.content
        result.blackboard_count = self.blackboard.count()
        result.event_count = len(self.event_bus.all_events())
        result.total_latency_ms = round((time.time() - t0) * 1000, 1)
        result.status = self._derive_status(subtask_results)

        self.decision_log.log(
            task_id=result.task_id,
            decision="workflow_complete",
            rationale=f"完成 {len(subtasks)} 个子任务，状态 {result.status}",
            context={"events": result.event_count, "mode": result.mode, "runtime": self.runtime_config},
        )

        # Phase 3: Record completion in observability
        status_tag = ObsEventType.TASK_COMPLETED if result.status == "completed" else ObsEventType.TASK_FAILED
        self.event_tracer.record(
            status_tag,
            task_id=result.task_id,
            duration_ms=result.total_latency_ms,
            data={"status": result.status, "mode": result.mode},
            message=f"Workflow {result.status} in {result.total_latency_ms}ms",
        )

        # Close streaming session
        if self.stream_session and not self.stream_session.producer.is_closed:
            self.stream_session.close()

        return result

    # ── DAG Subtask Execution ────────────────────────────────────────────

    def _execute_subtasks_dag(
        self,
        result: WorkflowResult,
        subtasks: list[dict],
        assignments: dict[str, str],
        mode: str,
        max_retries: int,
        t0: float,
    ) -> dict[str, dict]:
        """
        Execute subtasks with true DAG scheduling.

        Independent subtasks run in parallel via ThreadPoolExecutor.
        Dependent subtasks wait for their dependencies to complete first.
        Falls back to serial execution if the DAG has cycles.
        """
        n = len(subtasks)
        if n == 0:
            return {}

        node_ids = [f"{result.task_id}-T{idx:02d}" for idx in range(n)]

        # Map task_name → index for dependency resolution
        name_to_idx: dict[str, int] = {}
        for idx, sub in enumerate(subtasks):
            name = sub.get("task_name", "")
            if name:
                name_to_idx[name] = idx

        # Build adjacency and in-degree
        in_degree = [0] * n
        dependents: list[list[int]] = [[] for _ in range(n)]

        for idx, sub in enumerate(subtasks):
            deps = sub.get("depends_on", [])
            for dep in deps:
                if isinstance(dep, int) and 0 <= dep < n:
                    dep_idx = dep
                elif isinstance(dep, str) and dep in name_to_idx:
                    dep_idx = name_to_idx[dep]
                else:
                    continue
                if dep_idx != idx:
                    in_degree[idx] += 1
                    dependents[dep_idx].append(idx)

        # Topological sort into layers
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        layers: list[list[int]] = []
        visited = 0

        while queue:
            current_layer = list(queue)
            queue.clear()
            layers.append(current_layer)
            for node_idx in current_layer:
                visited += 1
                for dep_idx in dependents[node_idx]:
                    in_degree[dep_idx] -= 1
                    if in_degree[dep_idx] == 0:
                        queue.append(dep_idx)

        # ── P1-1: Build explicit task_name → task_id index (before cycle check) ──
        task_name_to_id: dict[str, str] = {}
        for idx, sub in enumerate(subtasks):
            name = sub.get("task_name", "")
            if name and name not in task_name_to_id:  # first wins on duplicates
                task_name_to_id[name] = node_ids[idx]

        # Fall back to serial if cycle detected
        if visited != n:
            return self._execute_subtasks_serial(
                result, subtasks, assignments, mode, max_retries, t0, node_ids, task_name_to_id,
            )

        subtask_results: dict[str, dict] = {}
        _lock = threading.Lock()

        for layer in layers:
            # Check cancellation before each layer
            if self._cancel_token and self._cancel_token.is_set():
                result.status = "cancelled"
                result.final_output = "工作流已取消。"
                result.total_latency_ms = round((time.time() - t0) * 1000, 1)
                break

            if len(layer) == 1:
                idx = layer[0]
                sub_id = node_ids[idx]
                subtask_results[sub_id] = self._execute_subtask(
                    workflow_id=result.task_id,
                    subtask_id=sub_id,
                    subtask=subtasks[idx],
                    assignments=assignments,
                    mode=mode,
                    max_retries=max_retries,
                    subtask_results=subtask_results,
                    task_name_to_id=task_name_to_id,
                )
                result.dag_nodes[idx]["status"] = self._derive_subtask_status(
                    subtask_results[sub_id]
                )
            else:
                # Multiple independent subtasks → parallel execution
                with ThreadPoolExecutor(max_workers=len(layer)) as executor:
                    future_map = {}
                    for idx in layer:
                        sub_id = node_ids[idx]
                        future = executor.submit(
                            self._execute_subtask,
                            workflow_id=result.task_id,
                            subtask_id=sub_id,
                            subtask=subtasks[idx],
                            assignments=assignments,
                            mode=mode,
                            max_retries=max_retries,
                            subtask_results=subtask_results,
                            task_name_to_id=task_name_to_id,
                        )
                        future_map[future] = (idx, sub_id)

                    for future in as_completed(future_map):
                        idx, sub_id = future_map[future]
                        try:
                            subtask_results[sub_id] = future.result()
                            result.dag_nodes[idx]["status"] = self._derive_subtask_status(
                                subtask_results[sub_id]
                            )
                        except Exception as e:
                            with _lock:
                                subtask_results[sub_id] = {
                                    "worker": AgentResponse(
                                        task_id=sub_id, role="Worker", status="failed"
                                    ),
                                    "final_content": f"错误: {e}",
                                }
                                result.dag_nodes[idx]["status"] = "failed"

        return subtask_results

    def _execute_subtasks_serial(
        self,
        result: WorkflowResult,
        subtasks: list[dict],
        assignments: dict[str, str],
        mode: str,
        max_retries: int,
        t0: float,
        node_ids: list[str],
        task_name_to_id: Optional[dict[str, str]] = None,
    ) -> dict[str, dict]:
        """Fallback serial execution when DAG has cycles."""
        subtask_results: dict[str, dict] = {}
        for idx, subtask in enumerate(subtasks):
            if self._cancel_token and self._cancel_token.is_set():
                result.status = "cancelled"
                result.final_output = "工作流已取消。"
                result.total_latency_ms = round((time.time() - t0) * 1000, 1)
                break

            sub_id = node_ids[idx] if idx < len(node_ids) else f"{result.task_id}-T{idx:02d}"
            subtask_results[sub_id] = self._execute_subtask(
                workflow_id=result.task_id,
                subtask_id=sub_id,
                subtask=subtask,
                assignments=assignments,
                mode=mode,
                max_retries=max_retries,
                subtask_results=subtask_results,
                task_name_to_id=task_name_to_id,
            )
            result.dag_nodes[idx]["status"] = self._derive_subtask_status(
                subtask_results[sub_id]
            )
        return subtask_results

    def _assign_roles(self, primary_model: str, complexity: ComplexityReport) -> dict[str, str]:
        available_models = self.adapter.available_models or [primary_model]
        if primary_model not in available_models:
            available_models = [primary_model] + available_models

        assignment_mode = self.runtime_config.get("assignment_mode", "primary_preferred")
        non_mock_models = [name for name in available_models if "mock" not in name.lower()]
        roles = ["Planner", "Worker", "Critic", "Verifier", "Supervisor"]
        if complexity.mode == MODE_EXPANDED:
            roles.extend(["Researcher", "Domain Expert"])

        # Phase 2: use ModelPool for healthy model filtering
        pool_healthy = [e.model_id for e in self.model_pool.healthy_models()]
        if pool_healthy:
            available_models = [m for m in available_models if m in pool_healthy] or available_models

        if assignment_mode == "single_model":
            assignments = {role: primary_model for role in roles}
        else:
            pool = non_mock_models or available_models
            assignments = self.role_assigner.assign(roles, pool if assignment_mode == "primary_preferred" else available_models)
            assignments["Planner"] = primary_model
            assignments["Supervisor"] = primary_model

        if len(available_models) == 1:
            assignments = {role: primary_model for role in roles}

        # Phase 2: apply RoleMergerSplitter merge/split/failover decisions
        self._apply_role_manager_decisions(assignments, roles, complexity)

        for role, model in assignments.items():
            self.event_bus.publish(
                EVENT_ROLE_ASSIGNED,
                source="DynamicRoleAssigner",
                payload={"role": role, "model": model},
            )
        return assignments

    def _execute_direct(
        self,
        result: WorkflowResult,
        user_task: str,
        model_name: str,
        complexity: ComplexityReport,
        started_at: float,
    ) -> WorkflowResult:
        # Phase 3: Check budget before direct execution
        if not self.budget_controller.can_call_model(result.task_id):
            result.status = "failed"
            result.final_output = "预算超限，无法执行。"
            result.total_latency_ms = round((time.time() - started_at) * 1000, 1)
            self.event_tracer.record(
                ObsEventType.BUDGET_EXCEEDED,
                task_id=result.task_id,
                message="Budget exceeded before direct execution",
            )
            return result

        self.event_tracer.record(
            ObsEventType.TASK_STARTED,
            task_id=result.task_id,
            data={"mode": "direct"},
            message="Direct mode execution started",
        )

        contract = TaskContract(
            task_id=f"{result.task_id}-DIRECT",
            task_name="直接执行",
            task_type="direct_answer",
            role="Worker",
            objective=user_task,
            assigned_model=model_name,
            required_capabilities=["reasoning"],
            allowed_tools=self.runtime_config.get("allowed_tools"),
            expected_output_schema="direct_result",
            success_criteria=["直接给出完整答案"],
            timeout_seconds=int(self.runtime_config.get("timeout_seconds", 120) or 120),
            review_required=False,
        )
        response = self.runner.run(contract, model_name, stream_session=self.stream_session)
        tool_records = self._execute_tool_calls(response, "Worker", contract.task_id, allowed_tools=contract.allowed_tools if contract.allowed_tools is not None else None)
        self._record("Worker", contract, response)
        result.subtask_results = {contract.task_id: {"worker": response, "final_content": response.content, "tool_records": tool_records, "contract": contract.to_dict()}}
        result.final_output = response.content
        result.blackboard_count = self.blackboard.count()
        result.total_latency_ms = round((time.time() - started_at) * 1000, 1)
        result.status = response.status if response.status else "completed"
        result.event_count = len(self.event_bus.all_events())
        self.decision_log.log(
            task_id=result.task_id,
            decision="direct_mode_complete",
            rationale=f"Direct mode answer by {model_name}",
            context={"complexity": complexity.__dict__, "runtime": self.runtime_config},
        )

        # Phase 3: Record completion
        status_tag = ObsEventType.TASK_COMPLETED if result.status == "completed" else ObsEventType.TASK_FAILED
        self.event_tracer.record(
            status_tag,
            task_id=result.task_id,
            duration_ms=result.total_latency_ms,
            data={"status": result.status, "mode": "direct"},
            message=f"Direct mode {result.status}",
        )

        # Close streaming session
        if self.stream_session and not self.stream_session.producer.is_closed:
            self.stream_session.close()

        return result

    def _plan_subtasks(self, workflow_id: str, user_task: str, model_name: str, complexity: ComplexityReport) -> list[dict]:
        count_hint = {
            MODE_SINGLE_REVIEW: "1~2",
            MODE_STANDARD: "3~4",
            MODE_EXPANDED: "4~5",
        }.get(complexity.mode, "2~3")
        contract = TaskContract(
            task_id=f"{workflow_id}-PLAN",
            task_name="任务规划",
            task_type="planning",
            role="Planner",
            objective=user_task,
            assigned_model=model_name,
            required_capabilities=["planning", "reasoning"],
            allowed_tools=self.runtime_config.get("allowed_tools"),
            expected_output_schema="subtasks",
            success_criteria=[f"生成 {count_hint} 个结构化的子任务"],
            timeout_seconds=int(self.runtime_config.get("timeout_seconds", 120) or 120),
        )
        context = [
            ChatMessage(
                role="user",
                content=(
                    f"## Complexity\n\n"
                    f"Mode: {complexity.mode}\n"
                    f"Score: {complexity.score}\n"
                    f"Rationale: {complexity.rationale}\n"
                ),
            )
        ]
        response = self.runner.run(contract, model_name, context_messages=context, stream_session=self.stream_session)
        self._record("Planner", contract, response)
        self.decision_log.log(
            task_id=workflow_id,
            decision="plan_created",
            rationale=f"Planner generated subtasks with {model_name}",
            context={"mode": complexity.mode, "runtime": self.runtime_config},
        )
        return self._parse_subtasks(response.content)

    def _collect_dependency_results(
        self,
        subtask: dict,
        subtask_results: dict,
        task_name_to_id: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """Collect upstream task results for dependency injection.

        Uses explicit task_name_to_id index for precise lookups, falling back
        to fuzzy matching only when the index does not contain a dependency name.
        Handles edge cases: duplicate names (first wins), missing deps (skipped),
        and upstream failures (included with failed status).
        """
        deps = subtask.get("depends_on", [])
        if not deps:
            return []

        # Build task_id_to_result from subtask_results for O(1) lookup
        task_id_to_result = {sid: sresult for sid, sresult in subtask_results.items()}

        results = []
        for dep_name in deps:
            target_id = None

            # 1. Try explicit name-to-id index first
            if task_name_to_id and dep_name in task_name_to_id:
                target_id = task_name_to_id[dep_name]

            # 2. Fallback: fuzzy match by task_id substring or worker task_id
            if target_id is None:
                for sid, sresult in subtask_results.items():
                    worker = sresult.get("worker")
                    worker_name = getattr(worker, "task_id", "") if worker else ""
                    if dep_name in sid or dep_name == worker_name:
                        target_id = sid
                        break

            # 3. Dependency not found — skip with warning in context
            if target_id is None or target_id not in task_id_to_result:
                continue

            sresult = task_id_to_result[target_id]
            worker = sresult.get("worker")
            results.append({
                "task_id": target_id,
                "final_content": sresult.get("final_content", ""),
                "status": getattr(worker, "status", "unknown") if worker else "unknown",
            })
        return results

    def _execute_subtask(
        self,
        workflow_id: str,
        subtask_id: str,
        subtask: dict,
        assignments: dict[str, str],
        mode: str,
        max_retries: int,
        subtask_results: dict = None,
        task_name_to_id: Optional[dict[str, str]] = None,
    ) -> dict:
        # Phase 3: Check budget before executing subtask
        if not self.budget_controller.can_call_model(workflow_id):
            return self._budget_exceeded_result(subtask_id, "模型调用预算已超限")

        worker_model = assignments["Worker"]
        critic_model = assignments["Critic"]
        verifier_model = assignments["Verifier"]
        researcher_model = assignments.get("Researcher", worker_model)
        expert_model = assignments.get("Domain Expert", worker_model)
        allowed_tools = self.runtime_config.get("allowed_tools")
        timeout_seconds = int(self.runtime_config.get("timeout_seconds", 120) or 120)

        # Phase 3: Build runtime context for this subtask
        dep_results = self._collect_dependency_results(
            subtask, subtask_results or {}, task_name_to_id=task_name_to_id,
        )
        self.context_manager.build_runtime_context(
            task_id=subtask_id,
            objective=subtask.get("objective", ""),
            dependencies=list(subtask.get("depends_on", [])),
        )
        self.context_manager.update_subtask_status(workflow_id, subtask_id, "running")

        # Inject upstream dependency results into context
        shared_dep_context: list[ChatMessage] = []
        for dr in dep_results:
            content_preview = dr["final_content"][:2000] if dr["final_content"] else ""
            shared_dep_context.append(ChatMessage(
                role="system",
                content=f"## 上游任务 {dr['task_id']} (状态: {dr['status']})\n\n{content_preview}",
            ))

        shared_context = list(shared_dep_context)
        if mode == MODE_EXPANDED:
            shared_context.extend(self._build_support_context(subtask_id, subtask, researcher_model, expert_model))

        worker_contract = TaskContract(
            task_id=subtask_id,
            task_name=subtask.get("task_name", subtask_id),
            task_type=subtask.get("task_type", "execution"),
            role="Worker",
            objective=subtask.get("objective", ""),
            input_refs=subtask.get("depends_on", []),
            assigned_model=worker_model,
            required_capabilities=["coding", "tool_use"],
            allowed_tools=allowed_tools,
            expected_output_schema="task_result",
            success_criteria=subtask.get("success_criteria", []),
            timeout_seconds=timeout_seconds,
        )
        # Phase 3: Worker execution with failover retry
        current_worker_model = worker_model
        worker_resp = self.runner.run(worker_contract, current_worker_model, context_messages=shared_context or None, stream_session=self.stream_session)
        worker_retries = 0
        while worker_resp.status == "failed" and worker_retries < max_retries:
            worker_retries += 1
            current_worker_model = self._attempt_failover(worker_model, subtask_id)
            if not current_worker_model:
                break
            worker_contract.assigned_model = current_worker_model
            worker_resp = self.runner.run(worker_contract, current_worker_model, context_messages=shared_context or None, stream_session=self.stream_session)

        self._execute_tool_calls(worker_resp, "Worker", subtask_id, allowed_tools=allowed_tools)
        self._record("Worker", worker_contract, worker_resp)
        current_content = worker_resp.content

        # P1-3: Track initial Worker output for Verifier context
        initial_worker_output = current_content

        if worker_resp.needs_review():
            decision = self.supervisor.event_handling.handle("low_confidence", {"role": "Worker", "subtask": subtask.get("task_name", "")})
            self._publish_replan_signal(workflow_id, subtask_id, worker_resp, "worker_low_confidence")

        # P1-3: Collect Critic findings across retries for Verifier context
        all_critic_findings: list[str] = []

        critic_resp = AgentResponse(task_id=subtask_id, role="Critic", status="completed")
        for retry in range(max_retries + 1):
            # P1-3: Critic TaskContract with full context
            critic_objective = (
                f"审查 Worker 输出质量。原始目标: {subtask.get('objective', '')}。"
                f"用户约束: {self.runtime_config.get('permission_mode', 'workspace_only')} 模式，"
                f"允许工具: {allowed_tools or []}。"
            )
            critic_contract = TaskContract(
                task_id=f"{subtask_id}-CRITIC",
                task_name=f"审查 {subtask.get('task_name', '')}",
                task_type="review",
                role="Critic",
                objective=critic_objective,
                input_refs=[subtask_id],
                assigned_model=critic_model,
                required_capabilities=["critic", "reasoning"],
                allowed_tools=allowed_tools,
                expected_output_schema="review_result",
                success_criteria=list(subtask.get("success_criteria", ["找出所有问题点"])),
                timeout_seconds=timeout_seconds,
            )
            # P1-3: Critic receives Worker output + upstream deps in context
            critic_context_messages: list[ChatMessage] = []
            if dep_results:
                for dr in dep_results:
                    content_preview = dr["final_content"][:2000] if dr["final_content"] else ""
                    critic_context_messages.append(ChatMessage(
                        role="system",
                        content=f"## 上游依赖 {dr['task_id']} (状态: {dr['status']})\n\n{content_preview}",
                    ))
            critic_context_messages.append(ChatMessage(role="user", content=f"## Worker 输出\n\n{current_content}"))
            critic_resp = self.runner.run(
                critic_contract,
                critic_model,
                context_messages=critic_context_messages or None,
                stream_session=self.stream_session,
            )
            self._execute_tool_calls(critic_resp, "Critic", f"{subtask_id}-CRITIC", allowed_tools=allowed_tools)
            self._record("Critic", critic_contract, critic_resp)
            verdict = (critic_resp.next_action_recommendation or "").lower()
            if "pass" in verdict:
                break

            # P1-3: Collect Critic findings for Verifier context
            if critic_resp.content:
                all_critic_findings.append(f"[第{retry + 1}轮审查] {critic_resp.content[:500]}")

            self.event_bus.publish(
                EVENT_REVIEW_FAILED,
                source="Critic",
                task_id=subtask_id,
                payload={"retry": retry, "summary": critic_resp.summary},
            )
            if retry >= max_retries:
                self._publish_replan_signal(workflow_id, subtask_id, critic_resp, "critic_exhausted")
                break

            revision_contract = TaskContract(
                task_id=f"{subtask_id}-REV{retry}",
                task_name=f"修订 {subtask.get('task_name', '')}",
                task_type="revision",
                role="Worker",
                objective="根据 Critic 意见修订输出",
                input_refs=[subtask_id],
                assigned_model=worker_model,
                required_capabilities=["coding", "reasoning"],
                allowed_tools=allowed_tools,
                expected_output_schema="task_result",
                success_criteria=["解决 Critic 指出的关键问题"],
                timeout_seconds=timeout_seconds,
            )
            revision_resp = self.runner.run(
                revision_contract,
                worker_model,
                context_messages=[
                    ChatMessage(role="user", content=f"## 批评意见\n\n{critic_resp.content}"),
                    ChatMessage(role="user", content=f"## 原始输出\n\n{current_content}"),
                ],
                stream_session=self.stream_session,
            )
            self._execute_tool_calls(revision_resp, "Worker", f"{subtask_id}-REV{retry}", allowed_tools=allowed_tools)
            self._record("Worker(revision)", revision_contract, revision_resp)
            current_content = revision_resp.content
            if revision_resp.needs_review():
                self._publish_replan_signal(workflow_id, subtask_id, revision_resp, "revision_low_confidence")

        # P1-3: Verifier TaskContract with complete context
        verifier_objective = (
            f"验证最终版本是否满足要求。原始目标: {subtask.get('objective', '')}。"
            f"需逐项验收成功标准。"
        )
        verifier_contract = TaskContract(
            task_id=f"{subtask_id}-VERIFY",
            task_name=f"校验 {subtask.get('task_name', '')}",
            task_type="verification",
            role="Verifier",
            objective=verifier_objective,
            input_refs=[subtask_id],
            assigned_model=verifier_model,
            required_capabilities=["verification", "reasoning"],
            allowed_tools=allowed_tools,
            expected_output_schema="verification_result",
            success_criteria=list(subtask.get("success_criteria", ["Critic 问题已解决"])),
            timeout_seconds=timeout_seconds,
        )
        # P1-3: Verifier receives full context — initial output, critic findings, final output
        verifier_context_messages: list[ChatMessage] = []
        verifier_context_messages.append(ChatMessage(
            role="system",
            content=f"## Worker 初稿\n\n{initial_worker_output[:2000]}",
        ))
        if all_critic_findings:
            verifier_context_messages.append(ChatMessage(
                role="system",
                content=f"## Critic 问题清单\n\n" + "\n---\n".join(all_critic_findings),
            ))
        verifier_context_messages.append(ChatMessage(
            role="user",
            content=f"## 最终输出\n\n{current_content}",
        ))
        verifier_resp = self.runner.run(
            verifier_contract,
            verifier_model,
            context_messages=verifier_context_messages or None,
            stream_session=self.stream_session,
        )
        self._execute_tool_calls(verifier_resp, "Verifier", f"{subtask_id}-VERIFY", allowed_tools=allowed_tools)
        self._record("Verifier", verifier_contract, verifier_resp)
        if verifier_resp.needs_review() or "fail" in (verifier_resp.next_action_recommendation or "").lower():
            self._publish_replan_signal(workflow_id, subtask_id, verifier_resp, "verifier_failed")

        self.artifact_store.save(
            artifact_id=f"ART-{subtask_id}",
            content=current_content,
            metadata={
                "task_name": subtask.get("task_name", ""),
                "worker_status": worker_resp.status,
                "verifier_status": verifier_resp.status,
                "worker_model": worker_model,
                "critic_model": critic_model,
                "verifier_model": verifier_model,
                "allowed_tools": allowed_tools,
                "workspace_root": self.runtime_config.get("workspace_root", ""),
                "permission_mode": self.runtime_config.get("permission_mode", "workspace_only"),
            },
        )
        return {
            "worker": worker_resp,
            "critic": critic_resp,
            "verifier": verifier_resp,
            "final_content": current_content,
            "contracts": {
                "worker": worker_contract.to_dict(),
                "critic": critic_contract.to_dict() if critic_resp else {},
                "verifier": verifier_contract.to_dict(),
            },
        }

    def _build_support_context(self, subtask_id: str, subtask: dict, researcher_model: str, expert_model: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        allowed_tools = self.runtime_config.get("allowed_tools")
        timeout_seconds = int(self.runtime_config.get("timeout_seconds", 120) or 120)

        researcher_contract = TaskContract(
            task_id=f"{subtask_id}-RESEARCH",
            task_name=f"研究 {subtask.get('task_name', '')}",
            task_type="research",
            role="Researcher",
            objective=f"为子任务提供实现资料和风险线索: {subtask.get('objective', '')}",
            assigned_model=researcher_model,
            required_capabilities=["research", "tool_use"],
            allowed_tools=allowed_tools,
            expected_output_schema="research_notes",
            success_criteria=["给出研究摘要"],
            timeout_seconds=timeout_seconds,
            review_required=False,
        )
        research_resp = self.runner.run(researcher_contract, researcher_model, stream_session=self.stream_session)
        self._record("Researcher", researcher_contract, research_resp)
        messages.append(ChatMessage(role="user", content=f"## Research Notes\n\n{research_resp.content}"))

        expert_contract = TaskContract(
            task_id=f"{subtask_id}-EXPERT",
            task_name=f"专家建议 {subtask.get('task_name', '')}",
            task_type="domain_review",
            role="Domain Expert",
            objective=f"从领域视角指出关键约束和风险: {subtask.get('objective', '')}",
            assigned_model=expert_model,
            required_capabilities=["domain_knowledge", "reasoning"],
            allowed_tools=allowed_tools,
            expected_output_schema="expert_notes",
            success_criteria=["给出领域约束"],
            timeout_seconds=timeout_seconds,
            review_required=False,
        )
        expert_resp = self.runner.run(expert_contract, expert_model, stream_session=self.stream_session)
        self._record("Domain Expert", expert_contract, expert_resp)
        messages.append(ChatMessage(role="user", content=f"## Domain Expert Notes\n\n{expert_resp.content}"))
        return messages

    def _build_synthesis_packet(self, subtask_id: str, subtask_result: dict) -> str:
        """Build structured summary for Supervisor synthesis."""
        content = subtask_result.get("final_content", "")
        worker = subtask_result.get("worker")

        # Get status and confidence from worker response
        status = getattr(worker, "status", "unknown") if worker else "unknown"
        confidence = getattr(worker, "confidence", None) if worker else None
        summary = getattr(worker, "summary", "") if worker else ""

        # Build structured packet
        packet = f"### 子任务 {subtask_id}\n"
        packet += f"**状态**: {status}"
        if confidence is not None:
            packet += f" | **置信度**: {confidence}"
        packet += "\n\n"

        # Include full summary from worker (typically concise)
        if summary:
            packet += f"**摘要**: {summary}\n\n"

        # Include full content if short enough, otherwise smart truncation
        if len(content) <= 1500:
            packet += content
        else:
            # Keep first 800 chars (usually contains key findings)
            packet += content[:800]
            packet += "\n\n...[内容已截断，详见子任务完整输出]..."
            # Always include last 400 chars (usually contains conclusions)
            last_part = content[-400:]
            if last_part not in content[:800]:
                packet += f"\n\n**结尾**: ...{last_part}"

        packet += "\n\n"

        # Include critic/verifier reports if available
        critic = subtask_result.get("critic")
        if critic and hasattr(critic, "to_dict"):
            cdict = critic.to_dict()
            issues = cdict.get("issues", [])
            if issues:
                packet += f"**审查意见**: {issues[:3]}\n\n"

        return packet

    def _summarize(self, workflow_id: str, subtask_results: dict, model_name: str) -> AgentResponse:
        summary_text = ""
        critic_reports: list[dict] = []
        verifier_reports: list[dict] = []
        for subtask_id, subtask_result in subtask_results.items():
            summary_text += self._build_synthesis_packet(subtask_id, subtask_result)
            critic_resp = subtask_result.get("critic")
            if critic_resp and hasattr(critic_resp, "to_dict"):
                critic_reports.append(critic_resp.to_dict())
            verifier_resp = subtask_result.get("verifier")
            if verifier_resp and hasattr(verifier_resp, "to_dict"):
                verifier_reports.append(verifier_resp.to_dict())

        contract = TaskContract(
            task_id=f"{workflow_id}-SUPER",
            task_name="最终汇总",
            task_type="summary",
            role="Supervisor",
            objective="综合所有子任务输出形成最终结果",
            assigned_model=model_name,
            required_capabilities=["synthesis", "decision"],
            allowed_tools=self.runtime_config.get("allowed_tools"),
            expected_output_schema="final_output",
            success_criteria=["覆盖全部子任务", "裁决冲突"],
            timeout_seconds=int(self.runtime_config.get("timeout_seconds", 120) or 120),
        )
        response = self.runner.run(
            contract,
            model_name,
            context_messages=[ChatMessage(role="user", content=f"## 所有子任务汇总\n\n{summary_text}")],
            stream_session=self.stream_session,
        )
        self._record("Supervisor", contract, response)

        # Phase 3: Supervisor final review
        review = self.supervisor.final_review.review(
            task_id=workflow_id,
            final_output=response.content,
            subtask_results=subtask_results,
            critic_reports=critic_reports,
            verifier_reports=verifier_reports,
        )
        self.decision_log.log(
            task_id=workflow_id,
            decision="final_review",
            rationale=f"Review: {review.status} (score={review.quality_score:.2f})",
            context={"review": review.to_dict()},
        )

        return response

    # ── Phase 3: Failover Helper ────────────────────────────────────────

    def _attempt_failover(self, failed_model: str, task_id: str) -> Optional[str]:
        """Try to find a replacement model when a model call fails."""
        # Record the failure and trigger role_manager failover
        affected = self.role_manager.handle_failover(failed_model)
        if not affected:
            return None

        # Find a healthy model from the pool
        healthy = self.model_pool.healthy_models()
        for entry in healthy:
            if entry.model_id != failed_model:
                self.decision_log.log(
                    task_id=task_id,
                    decision="model_failover_used",
                    rationale=f"Failover from {failed_model} to {entry.model_id}",
                    context={"original_model": failed_model, "new_model": entry.model_id},
                )
                return entry.model_id
        return None

    # ── Phase 2 helpers ─────────────────────────────────────────────

    def _ensure_pool_populated(self, primary_model: str) -> None:
        """Ensure all known models are registered in the ModelPool."""
        if self.model_pool.get(primary_model) is None:
            self.model_pool.register(primary_model, name=primary_model, provider="local")

        for model_name in (self.adapter.available_models or []):
            if self.model_pool.get(model_name) is None:
                self.model_pool.register(model_name, name=model_name, provider="local")

    def _apply_role_manager_decisions(
        self,
        assignments: dict[str, str],
        roles: list[str],
        complexity: ComplexityReport,
    ) -> None:
        """Apply merge/split/failover decisions from RoleMergerSplitter."""
        # Initialize bindings for each role
        for role in roles:
            model = assignments.get(role, self.main_model)
            if not self.role_manager.get_binding(role):
                self.role_manager.set_binding(
                    RoleBinding(
                        role=role,
                        primary_model=model,
                    )
                )

        # Attempt merges on adjacent roles when pool is small
        if self.model_pool.healthy_count <= 2:
            for i in range(len(roles) - 1):
                role_a = roles[i]
                role_b = roles[i + 1]
                if assignments.get(role_a) != assignments.get(role_b):
                    binding = self.role_manager.merge_roles(role_a, role_b)
                    if binding:
                        assignments[role_a] = binding.primary_model
                        assignments[role_b] = binding.primary_model

        # Attempt splits on heavily loaded roles when pool is large
        if self.model_pool.healthy_count >= 4:
            for role in roles:
                if self.role_manager.should_split(role):
                    self.role_manager.split_role(role)

    def _record_model_call_success(self, model_id: str) -> None:
        """Record a successful model call for health tracking."""
        self.model_pool.record_call_success(model_id)

    def _record_model_call_failure(self, model_id: str, error_message: str = "") -> None:
        """Record a failed model call and trigger failover if needed."""
        self.model_pool.record_call_failure(model_id, error_message)
        affected = self.role_manager.handle_failover(model_id)
        if affected:
            self.decision_log.log(
                task_id="failover",
                decision="model_failover_triggered",
                rationale=f"Model {model_id} failed, affected roles: {affected}",
                context={"model_id": model_id, "affected_roles": affected},
            )

    def _publish_replan_signal(self, workflow_id: str, subtask_id: str, response: AgentResponse, reason: str):
        event_type = EVENT_LOW_CONFIDENCE if response.status == "low_confidence" else EVENT_NEED_REPLAN
        event = self.event_bus.publish(
            event_type,
            source=response.role,
            task_id=subtask_id,
            payload={"reason": reason, "summary": response.summary, "recommendation": response.next_action_recommendation},
        )
        self.decision_log.log(
            task_id=workflow_id,
            decision="replan_requested",
            rationale=f"{response.role} emitted {event.event_type} for {subtask_id}",
            context=event.to_dict(),
        )

    def _derive_status(self, subtask_results: dict) -> str:
        statuses = []
        for item in subtask_results.values():
            statuses.extend([
                getattr(item.get("worker"), "status", "completed"),
                getattr(item.get("critic"), "status", "completed"),
                getattr(item.get("verifier"), "status", "completed"),
            ])
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status in ("need_review", "low_confidence") for status in statuses):
            return "partial"
        return "completed"

    def _derive_subtask_status(self, subtask_result: dict) -> str:
        statuses = [
            getattr(subtask_result.get("worker"), "status", "completed"),
            getattr(subtask_result.get("critic"), "status", "completed"),
            getattr(subtask_result.get("verifier"), "status", "completed"),
        ]
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status in ("need_review", "low_confidence") for status in statuses):
            return "partial"
        return "completed"

    def _record(self, role_label: str, contract: TaskContract, response: AgentResponse):
        entry = BlackboardEntry(
            task_id=contract.task_id,
            source_role=role_label,
            content_type="task_output",
            summary=response.summary[:100],
            payload={"contract": contract.to_dict(), "response": response.to_dict()},
        )
        self.blackboard.write(entry)

        # Phase 2: track model health per call
        model_id = contract.assigned_model
        if response.status in ("completed",):
            self._record_model_call_success(model_id)
        else:
            self._record_model_call_failure(model_id, error_message=response.summary[:200])

        # Phase 3: track budget per model call (use workflow-level budget)
        budget_task_id = getattr(self, '_current_workflow_id', None) or contract.task_id
        # Estimate tokens from response content length (~4 chars per token average)
        estimated_tokens = len(response.content) // 4 if response.content else 0
        # Rough cost estimate (assume $0.001 per 1K tokens as default)
        estimated_cost = estimated_tokens * 0.001 / 1000
        is_high = estimated_tokens > 10000

        self.budget_controller.record_model_call(
            task_id=budget_task_id,
            tokens=estimated_tokens,
            cost=estimated_cost,
            is_high_cost=is_high,
        )
        self.event_tracer.record(
            ObsEventType.TOOL_CALLED,
            task_id=contract.task_id,
            role=role_label,
            data={"model": model_id, "status": response.status},
            message=f"Model call: {role_label} on {model_id}",
        )

    def _execute_tool_calls(
        self,
        response: AgentResponse,
        role: str = "",
        task_id: str = "",
        allowed_tools: Optional[list[str]] = None,
    ) -> list[dict]:
        """Execute tool calls from an AgentResponse and return execution records."""
        if not response.has_tool_calls:
            return []
        records = self.tool_executor.execute_calls(
            response,
            role=role or response.role,
            task_id=task_id,
            allowed_tools=allowed_tools,
        )
        # Publish tool execution events
        for record in records:
            rd = record.to_dict()
            self.event_bus.publish(
                "TOOL_EXECUTED",
                source=rd.get("role", ""),
                task_id=rd.get("task_id", ""),
                payload={
                    "tool_name": rd.get("tool_name", ""),
                    "success": rd.get("success", False),
                    "permission_allowed": rd.get("permission_allowed", True),
                    "permission_reason": rd.get("permission_reason", ""),
                    "error": rd.get("error", ""),
                },
            )
            # Phase 3: track tool call in budget and observability
            budget_task_id = getattr(self, '_current_workflow_id', None) or task_id
            self.budget_controller.record_tool_call(task_id=budget_task_id)
            success_tag = rd.get("success", False)
            if not success_tag:
                self.event_tracer.record(
                    ObsEventType.TOOL_FAILED,
                    task_id=rd.get("task_id", task_id),
                    role=role or response.role,
                    data={"tool_name": rd.get("tool_name", ""), "error": rd.get("error", "")},
                    message=f"Tool failed: {rd.get('tool_name', '')}",
                )
        return [r.to_dict() for r in records]

    # ── Phase 3: Budget Helpers ──────────────────────────────────────────

    def _budget_exceeded_result(self, subtask_id: str, reason: str) -> dict:
        """Return a failed subtask result when budget is exceeded."""
        resp = AgentResponse(task_id=subtask_id, role="Worker", status="failed")
        resp.content = reason
        self.event_bus.publish(
            "BUDGET_EXCEEDED",
            source="BudgetController",
            task_id=subtask_id,
            payload={"reason": reason},
        )
        return {
            "worker": resp,
            "critic": AgentResponse(task_id=subtask_id, role="Critic", status="skipped"),
            "verifier": AgentResponse(task_id=subtask_id, role="Verifier", status="skipped"),
            "final_content": reason,
            "contracts": {},
        }

    @staticmethod
    def _parse_subtasks(content: str) -> list[dict]:
        import re

        json_str = None
        match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                json_str = match.group(1)
        if not json_str:
            return []
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []
        return data.get("subtasks", [])
