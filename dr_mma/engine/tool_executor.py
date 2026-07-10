"""
Tool Executor — 工具执行器。

桥接 ToolRegistry 和 PermissionManager，在工作流中实际执行模型的 tool_calls，
包含权限拦截、安全等级检查、审计日志和结果回传。

核心功能：
- 从 AgentResponse.tool_calls 提取工具调用请求
- 按 PermissionManager 检查角色权限和操作风险等级
- 通过 ToolRegistry 执行工具
- 收集 ToolResult 并附加到 AgentResponse.tool_results
- 维护完整的审计日志（权限拒绝、执行成功/失败）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .permissions import ActionLevel, PermissionManager
from .tools import ToolRegistry, ToolResult
from ..schemas.agent_response import AgentResponse, ToolCall


@dataclass
class ToolExecutionRecord:
    """单次工具执行的完整记录。"""

    tool_name: str = ""
    task_id: str = ""
    role: str = ""
    args: dict = field(default_factory=dict)
    result: Optional[ToolResult] = None
    permission_allowed: bool = True
    permission_reason: str = ""
    action_level: str = "safe"
    executed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "task_id": self.task_id,
            "role": self.role,
            "args": {k: str(v)[:200] for k, v in self.args.items()},
            "success": self.result.success if self.result else False,
            "output": self.result.output if self.result else None,
            "error": self.result.error if self.result else (
                self.permission_reason if not self.permission_allowed else ""
            ),
            "latency_ms": self.result.latency_ms if self.result else 0.0,
            "permission_allowed": self.permission_allowed,
            "permission_reason": self.permission_reason,
            "action_level": self.action_level,
            "executed_at": self.executed_at,
        }


class ToolExecutor:
    """
    工具执行器：在工作流中桥接工具调用和权限检查。

    Usage:
        executor = ToolExecutor(tool_registry, permission_manager)

        # 执行 AgentResponse 中的所有 tool_calls
        executor.execute_calls(response, role="Worker", task_id="T1")

        # 获取执行记录
        records = executor.get_records(task_id="T1")

        # 获取权限审计摘要
        summary = executor.audit_summary()
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        permission_manager: Optional[PermissionManager] = None,
    ):
        self.registry = registry or ToolRegistry()
        self.permission_manager = permission_manager or PermissionManager()
        self._records: list[ToolExecutionRecord] = []

    # ── 核心执行 ────────────────────────────────────────────────────

    def execute_calls(
        self,
        response: AgentResponse,
        role: str = "",
        task_id: str = "",
        allowed_tools: Optional[list[str]] = None,
    ) -> list[ToolExecutionRecord]:
        """
        执行 AgentResponse 中的所有 tool_calls。

        流程：
        1. 过滤不在 allowed_tools 中的调用
        2. 权限检查（角色权限 + 操作风险等级）
        3. 执行通过检查的工具调用
        4. 将结果附加到 response.tool_results

        Args:
            response: 包含 tool_calls 的 AgentResponse
            role: 当前执行角色
            task_id: 任务 ID
            allowed_tools: 允许的工具名称列表（None 表示全部允许）

        Returns:
            执行记录列表
        """
        if not response.has_tool_calls:
            return []

        records = []
        tool_results = list(response.tool_results)

        for tc in response.tool_calls:
            record = self._execute_single(
                tool_call=tc,
                role=role,
                task_id=task_id,
                allowed_tools=allowed_tools,
            )
            records.append(record)
            tool_results.append(record.to_dict())

        # 将结果写回 response
        response.tool_results = tool_results
        self._records.extend(records)
        return records

    def _execute_single(
        self,
        tool_call: ToolCall,
        role: str,
        task_id: str,
        allowed_tools: Optional[list[str]] = None,
    ) -> ToolExecutionRecord:
        """执行单个工具调用，包含权限检查。"""
        record = ToolExecutionRecord(
            tool_name=tool_call.tool_name,
            task_id=task_id,
            role=role,
            args=dict(tool_call.args),
        )

        # 第一步：检查是否在允许的工具列表中
        # None = use default policy (skip this check)
        # []    = block all tools explicitly
        # ["x"] = allow only listed tools
        if allowed_tools is not None and tool_call.tool_name not in allowed_tools:
            record.permission_allowed = False
            record.permission_reason = (
                f"Tool '{tool_call.tool_name}' not in allowed list: {allowed_tools}"
            )
            record.action_level = "blocked"
            return record

        # 第二步：权限检查
        perm_result = self.permission_manager.check(
            role=role,
            action=tool_call.tool_name,
            task_id=task_id,
        )
        record.action_level = perm_result.level.value

        if not perm_result.allowed:
            record.permission_allowed = False
            record.permission_reason = perm_result.reason
            return record

        # 第三步：检查工具是否存在
        if not self.registry.tool_exists(tool_call.tool_name):
            record.permission_allowed = False
            record.permission_reason = f"Tool '{tool_call.tool_name}' not registered"
            record.action_level = "not_found"
            return record

        # 第四步：执行工具
        result = self.registry.call(
            name=tool_call.tool_name,
            args=tool_call.args,
            role=role,
            task_id=task_id,
        )
        record.result = result
        record.permission_reason = "approved" if result.success else f"execution_failed: {result.error}"

        return record

    # ── 批量执行（多轮工具调用循环）───────────────────────────────────

    def execute_loop(
        self,
        responses: list[AgentResponse],
        role: str = "",
        task_id: str = "",
        allowed_tools: Optional[list[str]] = None,
        max_rounds: int = 5,
    ) -> list[ToolExecutionRecord]:
        """
        对多个 AgentResponse 执行工具调用循环。

        适用于工作流中多角色（Worker → Critic → Verifier）的工具调用。

        Args:
            responses: AgentResponse 列表
            role: 默认角色（每个 response 有自己的 role）
            task_id: 任务 ID
            allowed_tools: 允许的工具列表
            max_rounds: 最大执行轮数（防止无限循环）

        Returns:
            所有执行记录
        """
        all_records = []
        for resp in responses:
            actual_role = resp.role or role
            records = self.execute_calls(resp, actual_role, task_id, allowed_tools)
            all_records.extend(records)
        return all_records

    # ── 查询与诊断 ──────────────────────────────────────────────────

    def get_records(
        self,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> list[ToolExecutionRecord]:
        """获取执行记录。"""
        results = self._records
        if task_id:
            results = [r for r in results if r.task_id == task_id]
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        if role:
            results = [r for r in results if r.role == role]
        return results

    def get_records_dict(
        self,
        task_id: Optional[str] = None,
    ) -> list[dict]:
        """获取执行记录的字典形式（用于 UI 展示）。"""
        return [r.to_dict() for r in self.get_records(task_id)]

    def audit_summary(self, task_id: Optional[str] = None) -> dict:
        """返回工具执行审计摘要。"""
        records = self.get_records(task_id)
        total = len(records)
        allowed = sum(1 for r in records if r.permission_allowed)
        denied = total - allowed
        executed = sum(1 for r in records if r.result is not None)
        succeeded = sum(1 for r in records if r.result and r.result.success)
        failed = executed - succeeded

        by_level = {"safe": 0, "risky": 0, "critical": 0, "blocked": 0, "not_found": 0}
        for r in records:
            by_level[r.action_level] = by_level.get(r.action_level, 0) + 1

        return {
            "total_calls": total,
            "permission_allowed": allowed,
            "permission_denied": denied,
            "executed": executed,
            "succeeded": succeeded,
            "failed": failed,
            "by_level": by_level,
            "task_id": task_id or "all",
        }

    def permission_audit(self) -> dict:
        """返回权限管理器审计摘要。"""
        return self.permission_manager.audit_summary()

    def clear_records(self):
        """清空执行记录。"""
        self._records.clear()
