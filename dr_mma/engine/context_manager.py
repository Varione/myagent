"""
Context Manager — 三级上下文管理。

Phase 3: 防止多 agent 系统上下文膨胀，明确哪些内容进入上下文、哪些只存储不注入。

三级分级：
- Runtime Context: 当前子任务必须的最小上下文（执行 agent 读取）
- Global Context: 完整任务 DAG、事件日志、决策日志（Supervisor 专用）
- Artifact Context: 大文件/中间代码/长文档（只保存引用，不直接注入）

核心原则：Agent 默认只读取与当前 Task Contract 直接相关的上下文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeContext:
    """执行 agent 的最小运行时上下文。"""

    task_id: str
    objective: str
    direct_dependency_results: list[str] = field(default_factory=list)
    relevant_blackboard_entries: list[str] = field(default_factory=list)
    current_contract_snippet: str = ""
    output_format: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "direct_dependency_results": self.direct_dependency_results,
            "relevant_blackboard_entries": self.relevant_blackboard_entries,
            "current_contract_snippet": self.current_contract_snippet,
            "output_format": self.output_format,
        }

    @property
    def estimated_token_count(self) -> int:
        """粗略估算上下文 token 数（按字符/4 估算）。"""
        total = (
            len(self.objective)
            + sum(len(r) for r in self.direct_dependency_results)
            + sum(len(e) for e in self.relevant_blackboard_entries)
            + len(self.current_contract_snippet)
            + len(self.output_format)
        )
        return total // 4


@dataclass
class GlobalContext:
    """Supervisor 专用的全局上下文。"""

    task_id: str
    dag_state: dict = field(default_factory=dict)
    all_subtask_statuses: dict[str, str] = field(default_factory=dict)
    event_log_summary: list[dict] = field(default_factory=list)
    decision_log_summary: list[dict] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    budget_status: str = "within_budget"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "dag_state": self.dag_state,
            "all_subtask_statuses": self.all_subtask_statuses,
            "event_log_summary": self.event_log_summary,
            "decision_log_summary": self.decision_log_summary,
            "key_risks": self.key_risks,
            "budget_status": self.budget_status,
        }


@dataclass
class ArtifactContext:
    """产物引用上下文——只保存引用，不直接注入。"""

    artifact_id: str
    summary: str = ""
    content_uri: str = ""
    checksum: str = ""
    version: int = 1
    created_by_role: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "summary": self.summary,
            "content_uri": self.content_uri,
            "checksum": self.checksum,
            "version": self.version,
            "created_by_role": self.created_by_role,
            "metadata": self.metadata,
        }

    @property
    def reference_string(self) -> str:
        """生成简短的引用字符串，用于注入上下文。"""
        return f"[{self.artifact_id}] v{self.version}: {self.summary[:100]}"


class ContextManager:
    """
    管理三级上下文隔离，防止上下文膨胀。

    Usage:
        cm = ContextManager(max_runtime_tokens=40_000)

        # 构建执行 agent 的运行时上下文
        rc = cm.build_runtime_context(
            task_id="T-001",
            objective="完成数据分析",
            dependencies=["BB-001", "BB-002"],
            contract_snippet="...",
        )

        # Supervisor 读取全局上下文
        gc = cm.get_global_context("WF-001")

        # 产物只保存引用
        cm.register_artifact(artifact_id="ART-001", summary="初版报告", ...)
    """

    def __init__(self, max_runtime_tokens: int = 40_000):
        self.max_runtime_tokens = max_runtime_tokens
        self._runtime_contexts: dict[str, RuntimeContext] = {}
        self._global_contexts: dict[str, GlobalContext] = {}
        self._artifacts: dict[str, ArtifactContext] = {}

    # ── Runtime Context ─────────────────────────────────────────────

    def build_runtime_context(
        self,
        task_id: str,
        objective: str,
        dependencies: Optional[list[str]] = None,
        contract_snippet: str = "",
        output_format: str = "",
        blackboard_entries: Optional[list[str]] = None,
    ) -> RuntimeContext:
        """构建执行 agent 的最小运行时上下文。"""
        ctx = RuntimeContext(
            task_id=task_id,
            objective=objective,
            direct_dependency_results=dependencies or [],
            relevant_blackboard_entries=blackboard_entries or [],
            current_contract_snippet=contract_snippet,
            output_format=output_format,
        )
        self._runtime_contexts[task_id] = ctx
        return ctx

    def get_runtime_context(self, task_id: str) -> Optional[RuntimeContext]:
        """获取某任务的运行时上下文。"""
        return self._runtime_contexts.get(task_id)

    def check_token_budget(self, task_id: str) -> bool:
        """检查运行时上下文是否超出 token 预算。"""
        ctx = self._runtime_contexts.get(task_id)
        if ctx is None:
            return True
        return ctx.estimated_token_count <= self.max_runtime_tokens

    def trim_runtime_context(self, task_id: str) -> RuntimeContext:
        """裁剪运行时上下文至 token 预算内（优先保留直接依赖）。"""
        ctx = self._runtime_contexts.get(task_id)
        if ctx is None or self.check_token_budget(task_id):
            return ctx or RuntimeContext(task_id=task_id, objective="")

        # 先裁剪黑板条目（信息密度最低）
        while not self.check_token_budget(task_id) and ctx.relevant_blackboard_entries:
            ctx.relevant_blackboard_entries.pop(0)

        # 再裁剪间接依赖
        while not self.check_token_budget(task_id) and len(ctx.direct_dependency_results) > 1:
            ctx.direct_dependency_results.pop()

        return ctx

    # ── Global Context ──────────────────────────────────────────────

    def init_global_context(self, task_id: str) -> GlobalContext:
        """初始化全局上下文。"""
        ctx = GlobalContext(task_id=task_id)
        self._global_contexts[task_id] = ctx
        return ctx

    def get_global_context(self, task_id: str) -> Optional[GlobalContext]:
        """获取全局上下文（仅 Supervisor 应调用）。"""
        return self._global_contexts.get(task_id)

    def update_subtask_status(self, workflow_id: str, subtask_id: str, status: str):
        """更新子任务状态。"""
        gc = self._global_contexts.get(workflow_id)
        if gc:
            gc.all_subtask_statuses[subtask_id] = status

    def add_event_summary(self, workflow_id: str, event_summary: dict):
        """添加事件摘要到全局上下文。"""
        gc = self._global_contexts.get(workflow_id)
        if gc:
            gc.event_log_summary.append(event_summary)

    def add_decision_summary(self, workflow_id: str, decision_summary: dict):
        """添加决策摘要到全局上下文。"""
        gc = self._global_contexts.get(workflow_id)
        if gc:
            gc.decision_log_summary.append(decision_summary)

    def add_key_risk(self, workflow_id: str, risk: str):
        """添加关键风险到全局上下文。"""
        gc = self._global_contexts.get(workflow_id)
        if gc and risk not in gc.key_risks:
            gc.key_risks.append(risk)

    # ── Artifact Context ────────────────────────────────────────────

    def register_artifact(
        self,
        artifact_id: str,
        summary: str = "",
        content_uri: str = "",
        checksum: str = "",
        version: int = 1,
        created_by_role: str = "",
        metadata: Optional[dict] = None,
    ) -> ArtifactContext:
        """注册产物引用（不注入内容）。"""
        ctx = ArtifactContext(
            artifact_id=artifact_id,
            summary=summary,
            content_uri=content_uri,
            checksum=checksum,
            version=version,
            created_by_role=created_by_role,
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = ctx
        return ctx

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactContext]:
        """获取产物引用。"""
        return self._artifacts.get(artifact_id)

    def list_artifacts(self) -> list[ArtifactContext]:
        """列出所有产物引用。"""
        return list(self._artifacts.values())

    def get_artifact_references(self, artifact_ids: list[str]) -> list[str]:
        """获取多个产物的引用字符串列表（用于注入上下文）。"""
        refs = []
        for aid in artifact_ids:
            art = self._artifacts.get(aid)
            if art:
                refs.append(art.reference_string)
        return refs

    # ── 诊断 ────────────────────────────────────────────────────────

    def context_summary(self, task_id: str) -> dict:
        """返回某任务的上下文统计摘要。"""
        rc = self._runtime_contexts.get(task_id)
        gc = self._global_contexts.get(task_id)
        return {
            "task_id": task_id,
            "has_runtime_context": rc is not None,
            "runtime_token_estimate": rc.estimated_token_count if rc else 0,
            "within_budget": self.check_token_budget(task_id) if rc else True,
            "has_global_context": gc is not None,
            "subtask_count": len(gc.all_subtask_statuses) if gc else 0,
            "event_count": len(gc.event_log_summary) if gc else 0,
            "artifact_count": len(self._artifacts),
        }
