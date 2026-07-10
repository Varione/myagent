"""
Supervisor Modules — Supervisor 防过载设计。

Phase 3: Supervisor 拆分为独立子模块，每个模块有专用 prompt 和输入范围，
避免主控模型上下文压力持续累积。

分阶段加载策略：
- 任务开始时 → TaskUnderstanding + DAGPlanning
- 任务执行中 → EventHandling（持续监听）
- 事件触发时 → Decision 或 RoleAssignment
- 汇总阶段 → FinalReview
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── 子模块状态 ─────────────────────────────────────────────────────

@dataclass
class TaskUnderstandingResult:
    """任务理解模块输出。"""

    task_id: str
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    output_requirements: list[str] = field(default_factory=list)
    estimated_complexity: int = 0
    risk_level: str = "medium"  # low | medium | high
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "constraints": self.constraints,
            "output_requirements": self.output_requirements,
            "estimated_complexity": self.estimated_complexity,
            "risk_level": self.risk_level,
            "metadata": self.metadata,
        }


@dataclass
class DAGPlan:
    """DAG 规划模块输出。"""

    task_id: str
    nodes: list[dict] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "nodes": self.nodes,
            "edges": [list(e) for e in self.edges],
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
        }


@dataclass
class EventHandlingDecision:
    """事件处理模块输出。"""

    event_type: str
    action: str = ""  # ignore | escalate | trigger_debate | replan | retry
    rationale: str = ""
    target_role: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "action": self.action,
            "rationale": self.rationale,
            "target_role": self.target_role,
            "payload": self.payload,
        }


@dataclass
class DecisionResult:
    """裁决模块输出。"""

    conflict_id: str
    ruling: str = ""
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    dissenting_opinions: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "ruling": self.ruling,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "dissenting_opinions": self.dissenting_opinions,
            "confidence": self.confidence,
        }


@dataclass
class FinalReviewResult:
    """最终审查模块输出。"""

    task_id: str
    status: str = "pending"  # pending | passed | conditionally_passed | failed
    quality_score: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    final_output: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "quality_score": self.quality_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "final_output": self.final_output,
        }


# ── 各子模块实现 ───────────────────────────────────────────────────

class TaskUnderstandingModule:
    """任务理解模块：解析任务目标，提取约束。"""

    def analyze(
        self,
        task_id: str,
        user_task: str,
        constraints: Optional[list[str]] = None,
        output_format: str = "",
    ) -> TaskUnderstandingResult:
        """分析任务，提取结构化信息。"""
        result = TaskUnderstandingResult(task_id=task_id, goal=user_task)

        # 基础约束解析
        if constraints:
            result.constraints.extend(constraints)

        # 输出要求
        if output_format:
            result.output_requirements.append(f"按格式输出: {output_format}")

        # 复杂度估算（基于任务描述长度和关键词）
        complexity_keywords = ["分析", "设计", "实现", "评估", "比较", "优化"]
        keyword_count = sum(1 for kw in complexity_keywords if kw in user_task)
        result.estimated_complexity = max(1, min(12, len(user_task) // 20 + keyword_count * 2))

        # 风险等级判断
        risk_keywords = ["安全", "敏感", "关键", "紧急", "生产"]
        if any(kw in user_task for kw in risk_keywords):
            result.risk_level = "high"
        elif len(user_task) > 100:
            result.risk_level = "medium"
        else:
            result.risk_level = "low"

        return result


class DAGPlanningModule:
    """DAG 规划模块：生成和调整任务 DAG。"""

    def plan(
        self,
        task_id: str,
        understanding: TaskUnderstandingResult,
        subtasks: list[dict],
    ) -> DAGPlan:
        """根据理解结果和子任务列表生成 DAG 计划。"""
        plan = DAGPlan(task_id=task_id)

        # 创建节点
        for i, sub in enumerate(subtasks):
            node = {
                "node_id": f"{task_id}-T{i:02d}",
                "name": sub.get("task_name", f"Task {i}"),
                "objective": sub.get("objective", ""),
                "status": "pending",
            }
            plan.nodes.append(node)

        # 创建边（基于 depends_on）
        for i, sub in enumerate(subtasks):
            deps = sub.get("depends_on", [])
            for dep in deps:
                if isinstance(dep, int) and dep < len(subtasks):
                    plan.edges.append((f"{task_id}-T{dep:02d}", f"{task_id}-T{i:02d}"))

        # 计算执行顺序（简单拓扑排序）
        plan.execution_order = [n["node_id"] for n in plan.nodes]

        # 识别可并行组（无依赖关系的任务）
        independent = []
        for i, sub in enumerate(subtasks):
            if not sub.get("depends_on"):
                independent.append(f"{task_id}-T{i:02d}")
        if len(independent) > 1:
            plan.parallel_groups.append(independent)

        return plan


class EventHandlingModule:
    """事件处理模块：处理事件总线信号。"""

    def __init__(self):
        self._event_history: list[dict] = []

    def handle(self, event_type: str, payload: dict) -> EventHandlingDecision:
        """根据事件类型做出处理决策。"""
        decision = EventHandlingDecision(event_type=event_type)

        if event_type == "low_confidence":
            decision.action = "escalate"
            decision.rationale = "模型置信度低，需要升级处理"
            decision.target_role = "Supervisor"

        elif event_type == "review_failed":
            retry_count = payload.get("retry", 0)
            if retry_count < 2:
                decision.action = "retry"
                decision.rationale = f"审查未通过，第 {retry_count + 1} 次重试"
                decision.target_role = "Worker"
            else:
                decision.action = "trigger_debate"
                decision.rationale = "多次审查失败，触发受控讨论"
                decision.target_role = "DebateRoom"

        elif event_type == "need_replan":
            decision.action = "replan"
            decision.rationale = "需要重新规划任务 DAG"
            decision.target_role = "Planner"

        elif event_type == "tool_failed":
            decision.action = "retry"
            decision.rationale = "工具调用失败，尝试重试或降级"
            decision.target_role = "Worker"

        elif event_type == "conflict_detected":
            decision.action = "trigger_debate"
            decision.rationale = "检测到冲突，启动受控讨论"
            decision.target_role = "DebateRoom"

        else:
            decision.action = "ignore"
            decision.rationale = f"未知事件类型: {event_type}"

        self._event_history.append({
            "event_type": event_type,
            "action": decision.action,
            "timestamp": time.time(),
        })

        return decision

    def get_event_history(self) -> list[dict]:
        """获取事件处理历史。"""
        return list(self._event_history)


class DecisionModule:
    """裁决模块：裁决冲突和争议。"""

    def __init__(self):
        self._decisions: list[DecisionResult] = []

    def decide(
        self,
        conflict_id: str,
        conflict_type: str,
        positions: list[dict],
        evidence: Optional[list[str]] = None,
    ) -> DecisionResult:
        """
        对冲突做出裁决。

        Args:
            conflict_id: 冲突 ID
            conflict_type: 冲突类型 (fact | logic | plan | value | constraint)
            positions: 各方立场 [{"role": "Worker", "position": "..."}, ...]
            evidence: 证据引用列表
        """
        result = DecisionResult(
            conflict_id=conflict_id,
            evidence_refs=evidence or [],
        )

        # 裁决逻辑：基于证据和置信度
        if evidence:
            # 有证据支持 → 采用可验证的立场
            verifiable = [p for p in positions if p.get("has_evidence")]
            if verifiable:
                winner = max(verifiable, key=lambda p: p.get("confidence", 0))
                result.ruling = winner.get("position", "")
                result.rationale = f"基于证据裁决，采用 {winner.get('role', '')} 的立场"
                result.confidence = winner.get("confidence", 0.5)
            else:
                self._decide_by_consensus(conflict_id, positions, result)
        else:
            self._decide_by_consensus(conflict_id, positions, result)

        # 记录异议
        for pos in positions:
            if pos.get("position", "") != result.ruling:
                result.dissenting_opinions.append(
                    f"{pos.get('role', '')}: {pos.get('position', '')}"
                )

        self._decisions.append(result)
        return result

    def _decide_by_consensus(
        self,
        conflict_id: str,
        positions: list[dict],
        result: DecisionResult,
    ):
        """无证据时按共识裁决。"""
        if len(positions) <= 1:
            result.ruling = positions[0].get("position", "") if positions else ""
            result.rationale = "唯一立场，直接采用"
            return

        # 取最高置信度的立场
        winner = max(positions, key=lambda p: p.get("confidence", 0))
        result.ruling = winner.get("position", "")
        result.rationale = f"无明确证据，按最高置信度裁决（{winner.get('role', '')}）"
        result.confidence = winner.get("confidence", 0.5)

    def get_decisions(self) -> list[DecisionResult]:
        """获取所有裁决记录。"""
        return list(self._decisions)


class FinalReviewModule:
    """最终审查模块：审定输出质量。"""

    def review(
        self,
        task_id: str,
        final_output: str,
        subtask_results: dict,
        critic_reports: list[dict],
        verifier_reports: list[dict],
    ) -> FinalReviewResult:
        """综合审查最终输出质量。"""
        result = FinalReviewResult(
            task_id=task_id,
            final_output=final_output,
        )

        # 基于子任务状态评分
        total = len(subtask_results)
        completed = sum(1 for r in subtask_results.values() if getattr(r.get("worker"), "status", "") == "completed")
        result.quality_score = completed / total if total > 0 else 0

        # 基于审查报告判断
        critic_issues = []
        for report in critic_reports:
            verdict = (report.get("next_action_recommendation") or "").lower()
            if "pass" not in verdict:
                critic_issues.append(report.get("summary", ""))

        verifier_passed = all(
            "pass" in (r.get("next_action_recommendation") or "").lower()
            for r in verifier_reports
        ) if verifier_reports else True

        # 综合判定
        if result.quality_score >= 0.8 and verifier_passed and not critic_issues:
            result.status = "passed"
            result.strengths.append("所有子任务完成")
            result.strengths.append("审查通过")
        elif result.quality_score >= 0.5 and not critic_issues:
            result.status = "conditionally_passed"
            result.strengths.append("大部分子任务完成")
            result.weaknesses.append("部分子任务未完全满足要求")
        else:
            result.status = "failed"
            if critic_issues:
                result.weaknesses.extend(critic_issues[:3])
            if result.quality_score < 0.5:
                result.weaknesses.append("子任务完成率低于 50%")

        return result


# ── Supervisor 编排器 ───────────────────────────────────────────────

class SupervisorOrchestrator:
    """
    Supervisor 编排器：统一管理各子模块，支持分阶段加载。

    Usage:
        sup = SupervisorOrchestrator()

        # 第一阶段：任务理解 + DAG 规划
        understanding = sup.understand(task_id, user_task)
        dag_plan = sup.plan(understanding, subtasks)

        # 第二阶段：事件处理（持续）
        decision = sup.handle_event("low_confidence", {"source": "Worker"})

        # 第三阶段：裁决
        ruling = sup.decide(conflict_id, conflict_type, positions)

        # 第四阶段：最终审查
        review = sup.review(task_id, final_output, subtask_results, critics, verifiers)
    """

    def __init__(self):
        self.understanding = TaskUnderstandingModule()
        self.planning = DAGPlanningModule()
        self.event_handling = EventHandlingModule()
        self.decision = DecisionModule()
        self.final_review = FinalReviewModule()

        self._active_modules: set[str] = set()
        self._stage: str = "idle"

    def enter_stage(self, stage: str):
        """进入指定阶段，加载对应模块。"""
        self._stage = stage
        module_map = {
            "init": ["understanding", "planning"],
            "executing": ["event_handling"],
            "resolving": ["decision"],
            "reviewing": ["final_review"],
        }
        self._active_modules = set(module_map.get(stage, []))

    def is_module_active(self, module_name: str) -> bool:
        """检查模块是否在当前阶段激活。"""
        return module_name in self._active_modules

    @property
    def current_stage(self) -> str:
        return self._stage

    # ── 便捷方法 ─────────────────────────────────────────────────────

    def understand(self, task_id: str, user_task: str, **kwargs) -> TaskUnderstandingResult:
        """任务理解（第一阶段）。"""
        return self.understanding.analyze(task_id, user_task, **kwargs)

    def plan(self, understanding: TaskUnderstandingResult, subtasks: list[dict]) -> DAGPlan:
        """DAG 规划（第一阶段）。"""
        return self.planning.plan(understanding.task_id, understanding, subtasks)

    def handle_event(self, event_type: str, payload: dict) -> EventHandlingDecision:
        """事件处理（第二阶段）。"""
        return self.event_handling.handle(event_type, payload)

    def decide(self, conflict_id: str, conflict_type: str, positions: list[dict], **kwargs) -> DecisionResult:
        """裁决（第三阶段）。"""
        return self.decision.decide(conflict_id, conflict_type, positions, **kwargs)

    def review(
        self,
        task_id: str,
        final_output: str,
        subtask_results: dict,
        critic_reports: list[dict],
        verifier_reports: list[dict],
    ) -> FinalReviewResult:
        """最终审查（第四阶段）。"""
        return self.final_review.review(
            task_id, final_output, subtask_results, critic_reports, verifier_reports
        )
