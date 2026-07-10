"""Workflow engine with complexity routing, dynamic roles, and event logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Optional

from ..models.adapter import ChatMessage, ModelAdapter
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
from .model_pool import ModelPool
from .role_manager import RoleBinding, RoleMergerSplitter


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

    def execute(self, user_task: str, model_name: str = "", max_retries: int = 1) -> WorkflowResult:
        import time

        t0 = time.time()
        primary_model = model_name or self.main_model
        if not primary_model:
            raise ValueError("必须指定模型名称")

        # Phase 2: ensure all models are registered in the pool and health-checked
        self._ensure_pool_populated(primary_model)
        self.model_pool.health_check_all()

        result = WorkflowResult(task_id=f"WF-{hash(user_task) % 100000:05d}")
        result.runtime_config = dict(self.runtime_config)
        complexity = self.complexity_evaluator.evaluate(user_task)
        assignments = self._assign_roles(primary_model, complexity)
        result.mode = complexity.mode
        result.complexity_score = complexity.score
        result.role_assignments = assignments

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
            result.status = "failed"
            result.final_output = "规划阶段未能生成有效子任务。"
            result.total_latency_ms = round((time.time() - t0) * 1000, 1)
            return result

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

        subtask_results = {}
        for idx, subtask in enumerate(subtasks):
            sub_id = f"{result.task_id}-T{idx:02d}"
            subtask_results[sub_id] = self._execute_subtask(
                workflow_id=result.task_id,
                subtask_id=sub_id,
                subtask=subtask,
                assignments=assignments,
                mode=complexity.mode,
                max_retries=max_retries,
            )
            result.dag_nodes[idx]["status"] = self._derive_subtask_status(subtask_results[sub_id])

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
        return result

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
        contract = TaskContract(
            task_id=f"{result.task_id}-DIRECT",
            task_name="直接执行",
            task_type="direct_answer",
            role="Worker",
            objective=user_task,
            assigned_model=model_name,
            required_capabilities=["reasoning"],
            allowed_tools=list(self.runtime_config.get("allowed_tools", [])),
            expected_output_schema="direct_result",
            success_criteria=["直接给出完整答案"],
            timeout_seconds=int(self.runtime_config.get("timeout_seconds", 120) or 120),
            review_required=False,
        )
        response = self.runner.run(contract, model_name)
        self._record("Worker", contract, response)
        result.subtask_results = {contract.task_id: {"worker": response, "final_content": response.content, "contract": contract.to_dict()}}
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
            allowed_tools=list(self.runtime_config.get("allowed_tools", [])),
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
        response = self.runner.run(contract, model_name, context_messages=context)
        self._record("Planner", contract, response)
        self.decision_log.log(
            task_id=workflow_id,
            decision="plan_created",
            rationale=f"Planner generated subtasks with {model_name}",
            context={"mode": complexity.mode, "runtime": self.runtime_config},
        )
        return self._parse_subtasks(response.content)

    def _execute_subtask(
        self,
        workflow_id: str,
        subtask_id: str,
        subtask: dict,
        assignments: dict[str, str],
        mode: str,
        max_retries: int,
    ) -> dict:
        worker_model = assignments["Worker"]
        critic_model = assignments["Critic"]
        verifier_model = assignments["Verifier"]
        researcher_model = assignments.get("Researcher", worker_model)
        expert_model = assignments.get("Domain Expert", worker_model)
        allowed_tools = list(self.runtime_config.get("allowed_tools", []))
        timeout_seconds = int(self.runtime_config.get("timeout_seconds", 120) or 120)

        shared_context = []
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
        worker_resp = self.runner.run(worker_contract, worker_model, context_messages=shared_context or None)
        self._record("Worker", worker_contract, worker_resp)
        current_content = worker_resp.content

        if worker_resp.needs_review():
            self._publish_replan_signal(workflow_id, subtask_id, worker_resp, "worker_low_confidence")

        critic_resp = AgentResponse(task_id=subtask_id, role="Critic", status="completed")
        for retry in range(max_retries + 1):
            critic_contract = TaskContract(
                task_id=f"{subtask_id}-CRITIC",
                task_name=f"审查 {subtask.get('task_name', '')}",
                task_type="review",
                role="Critic",
                objective="审查 Worker 输出质量",
                input_refs=[subtask_id],
                assigned_model=critic_model,
                required_capabilities=["critic", "reasoning"],
                allowed_tools=allowed_tools,
                expected_output_schema="review_result",
                success_criteria=["找出所有问题点"],
                timeout_seconds=timeout_seconds,
            )
            critic_resp = self.runner.run(
                critic_contract,
                critic_model,
                context_messages=[ChatMessage(role="user", content=f"## Worker 输出\n\n{current_content}")],
            )
            self._record("Critic", critic_contract, critic_resp)
            verdict = (critic_resp.next_action_recommendation or "").lower()
            if "pass" in verdict:
                break

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
            )
            self._record("Worker(revision)", revision_contract, revision_resp)
            current_content = revision_resp.content
            if revision_resp.needs_review():
                self._publish_replan_signal(workflow_id, subtask_id, revision_resp, "revision_low_confidence")

        verifier_contract = TaskContract(
            task_id=f"{subtask_id}-VERIFY",
            task_name=f"校验 {subtask.get('task_name', '')}",
            task_type="verification",
            role="Verifier",
            objective="验证最终版本是否满足要求",
            input_refs=[subtask_id],
            assigned_model=verifier_model,
            required_capabilities=["verification", "reasoning"],
            allowed_tools=allowed_tools,
            expected_output_schema="verification_result",
            success_criteria=["Critic 问题已解决"],
            timeout_seconds=timeout_seconds,
        )
        verifier_resp = self.runner.run(
            verifier_contract,
            verifier_model,
            context_messages=[ChatMessage(role="user", content=f"## 最终输出\n\n{current_content}")],
        )
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
        allowed_tools = list(self.runtime_config.get("allowed_tools", []))
        timeout_seconds = int(self.runtime_config.get("timeout_seconds", 120) or 120)

        researcher_contract = TaskContract(
            task_id=f"{subtask_id}-RESEARCH",
            task_name=f"研究 {subtask.get('task_name', '')}",
            task_type="research",
            role="Worker",
            objective=f"为子任务提供实现资料和风险线索: {subtask.get('objective', '')}",
            assigned_model=researcher_model,
            required_capabilities=["research", "tool_use"],
            allowed_tools=allowed_tools,
            expected_output_schema="research_notes",
            success_criteria=["给出研究摘要"],
            timeout_seconds=timeout_seconds,
            review_required=False,
        )
        research_resp = self.runner.run(researcher_contract, researcher_model)
        self._record("Researcher", researcher_contract, research_resp)
        messages.append(ChatMessage(role="user", content=f"## Research Notes\n\n{research_resp.content}"))

        expert_contract = TaskContract(
            task_id=f"{subtask_id}-EXPERT",
            task_name=f"专家建议 {subtask.get('task_name', '')}",
            task_type="domain_review",
            role="Worker",
            objective=f"从领域视角指出关键约束和风险: {subtask.get('objective', '')}",
            assigned_model=expert_model,
            required_capabilities=["domain_knowledge", "reasoning"],
            allowed_tools=allowed_tools,
            expected_output_schema="expert_notes",
            success_criteria=["给出领域约束"],
            timeout_seconds=timeout_seconds,
            review_required=False,
        )
        expert_resp = self.runner.run(expert_contract, expert_model)
        self._record("Domain Expert", expert_contract, expert_resp)
        messages.append(ChatMessage(role="user", content=f"## Domain Expert Notes\n\n{expert_resp.content}"))
        return messages

    def _summarize(self, workflow_id: str, subtask_results: dict, model_name: str) -> AgentResponse:
        summary_text = ""
        for subtask_id, subtask_result in subtask_results.items():
            summary_text += f"### 子任务 {subtask_id}\n{subtask_result['final_content'][:400]}\n\n"

        contract = TaskContract(
            task_id=f"{workflow_id}-SUPER",
            task_name="最终汇总",
            task_type="summary",
            role="Supervisor",
            objective="综合所有子任务输出形成最终结果",
            assigned_model=model_name,
            required_capabilities=["synthesis", "decision"],
            allowed_tools=list(self.runtime_config.get("allowed_tools", [])),
            expected_output_schema="final_output",
            success_criteria=["覆盖全部子任务", "裁决冲突"],
            timeout_seconds=int(self.runtime_config.get("timeout_seconds", 120) or 120),
        )
        response = self.runner.run(
            contract,
            model_name,
            context_messages=[ChatMessage(role="user", content=f"## 所有子任务汇总\n\n{summary_text}")],
        )
        self._record("Supervisor", contract, response)
        return response

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
