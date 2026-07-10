"""
Budget Controller — 预算控制器。

多模型协同系统中，Debate Room、Critic、Verifier、重试机制叠加后成本容易失控。
每个任务初始化时生成预算，追踪器实时监控消耗，进入警告状态时通知 Supervisor。

核心功能：
- 预算初始化（model_calls, tool_calls, tokens, retries, debates）
- 实时追踪消耗
- 三级状态转换（within_budget → warning → exceeded）
- 警告事件触发
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetConfig:
    """预算配置。"""

    task_id: str
    max_model_calls: int = 30
    max_tool_calls: int = 10
    max_debate_rounds: int = 2
    max_retries_per_node: int = 2
    max_total_tokens: int = 120_000
    allow_high_cost_model: bool = True
    budget_owner: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_debate_rounds": self.max_debate_rounds,
            "max_retries_per_node": self.max_retries_per_node,
            "max_total_tokens": self.max_total_tokens,
            "allow_high_cost_model": self.allow_high_cost_model,
            "budget_owner": self.budget_owner,
        }


@dataclass
class BudgetUsage:
    """预算使用追踪。"""

    task_id: str
    model_calls_used: int = 0
    tool_calls_used: int = 0
    debate_rounds_used: int = 0
    retries_used: dict[str, int] = field(default_factory=dict)
    tokens_consumed: int = 0
    high_cost_calls: int = 0
    estimated_cost: float = 0.0
    status: str = "within_budget"  # within_budget | warning | exceeded
    warnings_issued: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_calls_used": self.model_calls_used,
            "tool_calls_used": self.tool_calls_used,
            "debate_rounds_used": self.debate_rounds_used,
            "retries_used": self.retries_used,
            "tokens_consumed": self.tokens_consumed,
            "high_cost_calls": self.high_cost_calls,
            "estimated_cost": round(self.estimated_cost, 4),
            "status": self.status,
            "warnings_issued": self.warnings_issued,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class BudgetController:
    """
    预算控制器：监控多模型协同任务的资源消耗。

    Usage:
        bc = BudgetController()
        bc.initialize("TASK-001", max_model_calls=30, max_total_tokens=120_000)

        # 追踪消耗
        bc.record_model_call("TASK-001", tokens=4000, cost=0.05)
        bc.record_tool_call("TASK-001")
        bc.record_debate_round("TASK-001")
        bc.record_retry("TASK-001", "T-001")

        # 检查状态
        usage = bc.get_usage("TASK-001")
        if usage.status == "warning":
            # Supervisor 收到警告，决定是否追加预算或降级
            pass

        # 申请追加预算
        bc.increase_budget("TASK-001", extra_model_calls=10)
    """

    def __init__(self, warning_threshold: float = 0.8):
        self.warning_threshold = warning_threshold
        self._configs: dict[str, BudgetConfig] = {}
        self._usages: dict[str, BudgetUsage] = {}

    # ── 初始化 ──────────────────────────────────────────────────────

    def initialize(
        self,
        task_id: str,
        max_model_calls: int = 30,
        max_tool_calls: int = 10,
        max_debate_rounds: int = 2,
        max_retries_per_node: int = 2,
        max_total_tokens: int = 120_000,
        allow_high_cost_model: bool = True,
        budget_owner: str = "",
    ) -> BudgetConfig:
        """初始化任务预算。"""
        config = BudgetConfig(
            task_id=task_id,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            max_debate_rounds=max_debate_rounds,
            max_retries_per_node=max_retries_per_node,
            max_total_tokens=max_total_tokens,
            allow_high_cost_model=allow_high_cost_model,
            budget_owner=budget_owner,
        )
        self._configs[task_id] = config
        self._usages[task_id] = BudgetUsage(task_id=task_id)
        return config

    def get_config(self, task_id: str) -> Optional[BudgetConfig]:
        """获取预算配置。"""
        return self._configs.get(task_id)

    # ── 消耗追踪 ────────────────────────────────────────────────────

    def record_model_call(
        self,
        task_id: str,
        tokens: int = 0,
        cost: float = 0.0,
        is_high_cost: bool = False,
    ) -> BudgetUsage:
        """记录模型调用消耗。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return usage or BudgetUsage(task_id=task_id)

        usage.model_calls_used += 1
        usage.tokens_consumed += tokens
        usage.estimated_cost += cost
        if is_high_cost:
            usage.high_cost_calls += 1
        usage.last_updated = time.time()

        self._check_status(task_id)
        return usage

    def record_tool_call(self, task_id: str) -> BudgetUsage:
        """记录工具调用消耗。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return usage or BudgetUsage(task_id=task_id)

        usage.tool_calls_used += 1
        usage.last_updated = time.time()

        self._check_status(task_id)
        return usage

    def record_debate_round(self, task_id: str) -> BudgetUsage:
        """记录讨论轮次消耗。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return usage or BudgetUsage(task_id=task_id)

        usage.debate_rounds_used += 1
        usage.last_updated = time.time()

        self._check_status(task_id)
        return usage

    def record_retry(self, task_id: str, node_id: str) -> BudgetUsage:
        """记录重试消耗。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return usage or BudgetUsage(task_id=task_id)

        usage.retries_used[node_id] = usage.retries_used.get(node_id, 0) + 1
        usage.last_updated = time.time()

        self._check_status(task_id)
        return usage

    def record_tokens(self, task_id: str, tokens: int) -> BudgetUsage:
        """记录 Token 消耗（不增加调用次数）。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return usage or BudgetUsage(task_id=task_id)

        usage.tokens_consumed += tokens
        usage.last_updated = time.time()

        self._check_status(task_id)
        return usage

    # ── 状态检查 ────────────────────────────────────────────────────

    def _check_status(self, task_id: str):
        """内部：检查并更新预算状态。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)
        if config is None or usage is None:
            return

        # 计算各维度消耗比例
        ratios = []
        if config.max_model_calls > 0:
            ratios.append(usage.model_calls_used / config.max_model_calls)
        if config.max_tool_calls > 0:
            ratios.append(usage.tool_calls_used / config.max_tool_calls)
        if config.max_total_tokens > 0:
            ratios.append(usage.tokens_consumed / config.max_total_tokens)
        if config.max_debate_rounds > 0:
            ratios.append(usage.debate_rounds_used / config.max_debate_rounds)

        if not ratios:
            return

        max_ratio = max(ratios)
        exceeded_any = False

        # 检查是否任何维度已超出
        if (config.max_model_calls > 0 and usage.model_calls_used >= config.max_model_calls) or \
           (config.max_tool_calls > 0 and usage.tool_calls_used >= config.max_tool_calls) or \
           (config.max_total_tokens > 0 and usage.tokens_consumed >= config.max_total_tokens) or \
           (config.max_debate_rounds > 0 and usage.debate_rounds_used >= config.max_debate_rounds):
            exceeded_any = True

        # 检查重试是否超出
        for node_id, count in usage.retries_used.items():
            if count >= config.max_retries_per_node:
                exceeded_any = True
                break

        old_status = usage.status
        if exceeded_any:
            usage.status = "exceeded"
        elif max_ratio >= self.warning_threshold:
            usage.status = "warning"
        else:
            usage.status = "within_budget"

        # 状态变更时记录警告
        if old_status != usage.status and usage.status in ("warning", "exceeded"):
            usage.warnings_issued.append({
                "old_status": old_status,
                "new_status": usage.status,
                "timestamp": time.time(),
                "max_ratio": round(max_ratio, 3),
            })

    def get_usage(self, task_id: str) -> Optional[BudgetUsage]:
        """获取当前预算使用情况。"""
        return self._usages.get(task_id)

    def is_within_budget(self, task_id: str) -> bool:
        """检查是否在预算内。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return True
        return usage.status == "within_budget"

    def can_call_model(self, task_id: str) -> bool:
        """检查是否还能调用模型。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)
        if config is None or usage is None:
            return True
        return usage.model_calls_used < config.max_model_calls

    def can_call_tool(self, task_id: str) -> bool:
        """检查是否还能调用工具。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)
        if config is None or usage is None:
            return True
        return usage.tool_calls_used < config.max_tool_calls

    def can_debate(self, task_id: str) -> bool:
        """检查是否还能进行讨论。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)
        if config is None or usage is None:
            return True
        return usage.debate_rounds_used < config.max_debate_rounds

    def can_retry(self, task_id: str, node_id: str) -> bool:
        """检查某节点是否还能重试。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)
        if config is None or usage is None:
            return True
        count = usage.retries_used.get(node_id, 0)
        return count < config.max_retries_per_node

    # ── 预算调整 ────────────────────────────────────────────────────

    def increase_budget(
        self,
        task_id: str,
        extra_model_calls: int = 0,
        extra_tool_calls: int = 0,
        extra_tokens: int = 0,
        extra_debate_rounds: int = 0,
    ) -> Optional[BudgetConfig]:
        """追加预算。"""
        config = self._configs.get(task_id)
        if config is None:
            return None

        config.max_model_calls += extra_model_calls
        config.max_tool_calls += extra_tool_calls
        config.max_total_tokens += extra_tokens
        config.max_debate_rounds += extra_debate_rounds

        # 追加后重新检查状态（可能从 exceeded 回到 warning）
        self._check_status(task_id)
        return config

    def reset_usage(self, task_id: str) -> Optional[BudgetUsage]:
        """重置使用计数（用于新阶段）。"""
        usage = self._usages.get(task_id)
        if usage is None:
            return None

        usage.model_calls_used = 0
        usage.tool_calls_used = 0
        usage.debate_rounds_used = 0
        usage.retries_used.clear()
        usage.tokens_consumed = 0
        usage.high_cost_calls = 0
        usage.estimated_cost = 0.0
        usage.status = "within_budget"
        usage.warnings_issued.clear()
        usage.last_updated = time.time()
        return usage

    # ── 诊断 ────────────────────────────────────────────────────────

    def budget_summary(self, task_id: str) -> dict:
        """返回预算摘要。"""
        config = self._configs.get(task_id)
        usage = self._usages.get(task_id)

        if config is None or usage is None:
            return {"task_id": task_id, "error": "not initialized"}

        return {
            "task_id": task_id,
            "status": usage.status,
            "model_calls": f"{usage.model_calls_used}/{config.max_model_calls}",
            "tool_calls": f"{usage.tool_calls_used}/{config.max_tool_calls}",
            "debate_rounds": f"{usage.debate_rounds_used}/{config.max_debate_rounds}",
            "tokens": f"{usage.tokens_consumed}/{config.max_total_tokens}",
            "high_cost_calls": usage.high_cost_calls,
            "estimated_cost": round(usage.estimated_cost, 4),
            "warnings_count": len(usage.warnings_issued),
        }
