"""
Permissions — 权限安全层。

明确不同 agent 的权限边界，三级操作分级，高风险操作必须人工确认。

核心功能：
- 权限矩阵：角色 × 操作的允许/拒绝表
- 操作分级：Safe（自动）、Risky（需 Supervisor 审批）、Critical（需人工确认）
- 权限检查器：运行时验证 agent 是否有权执行某操作
- 审计日志：记录所有权限相关事件
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionLevel(Enum):
    """操作风险等级。"""

    SAFE = "safe"  # 自动执行
    RISKY = "risky"  # 需 Supervisor 审批
    CRITICAL = "critical"  # 需人工确认


# ── 标准权限矩阵（架构计划定义）──────────────────────────────────────

DEFAULT_PERMISSION_MATRIX: dict[str, set[str]] = {
    "Planner": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
    },
    "Executor": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
        "code_execute",
        "api_call",
    },
    "Researcher": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
        "web_search",
        "file_parse",
        "database_query",
    },
    "Critic": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
    },
    "Verifier": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
        "code_execute",
        "database_query",
    },
    "Supervisor": {
        "blackboard_read",
        "blackboard_write",
        "read_artifact",
        "code_execute",
        "api_call",
        "web_search",
        "file_parse",
        "database_query",
        "modify_dag",
        "assign_role",
        "trigger_debate",
        "approve_risky_action",
    },
}

# ── 操作风险分级 ────────────────────────────────────────────────────

ACTION_LEVELS: dict[str, ActionLevel] = {
    # Safe — 自动执行
    "blackboard_read": ActionLevel.SAFE,
    "blackboard_write": ActionLevel.SAFE,
    "read_artifact": ActionLevel.SAFE,
    "code_execute": ActionLevel.CRITICAL,
    "database_query": ActionLevel.SAFE,
    # Risky — 需 Supervisor 审批
    "api_call": ActionLevel.RISKY,
    "web_search": ActionLevel.RISKY,
    "file_parse": ActionLevel.RISKY,
    "modify_dag": ActionLevel.RISKY,
    "assign_role": ActionLevel.RISKY,
    "trigger_debate": ActionLevel.RISKY,
    # Critical — 需人工确认
    "delete_file": ActionLevel.CRITICAL,
    "send_external_message": ActionLevel.CRITICAL,
    "modify_system_config": ActionLevel.CRITICAL,
    "approve_risky_action": ActionLevel.SAFE,
}


@dataclass
class PermissionCheckResult:
    """权限检查结果。"""

    role: str
    action: str
    allowed: bool
    reason: str = ""
    level: ActionLevel = ActionLevel.SAFE
    requires_approval: str = ""  # none | supervisor | human

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "action": self.action,
            "allowed": self.allowed,
            "reason": self.reason,
            "level": self.level.value,
            "requires_approval": self.requires_approval,
        }


@dataclass
class AuditEntry:
    """审计日志条目。"""

    entry_id: str = ""
    timestamp: float = field(default_factory=time.time)
    role: str = ""
    action: str = ""
    task_id: str = ""
    allowed: bool = False
    reason: str = ""
    approval_by: str = ""  # auto | supervisor | human

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "action": self.action,
            "task_id": self.task_id,
            "allowed": self.allowed,
            "reason": self.reason,
            "approval_by": self.approval_by,
        }


class PermissionManager:
    """
    权限管理器：管理角色权限矩阵和操作分级。

    Usage:
        pm = PermissionManager(mode="workspace_only")

        # 检查权限
        result = pm.check("Worker", "code_execute", task_id="T1")
        if result.allowed:
            execute_code(...)

        # 获取操作等级
        level = pm.get_action_level("delete_file")
        assert level == ActionLevel.CRITICAL

        # 自定义权限矩阵
        pm.set_permissions("CustomRole", {"blackboard_read", "custom_action"})
    """

    def __init__(self, matrix: Optional[dict[str, set[str]]] = None, mode: str = "workspace_only"):
        import copy
        self._mode = mode
        self._matrix = copy.deepcopy(matrix) if matrix is not None else copy.deepcopy(DEFAULT_PERMISSION_MATRIX)
        self._audit_log: list[AuditEntry] = []
        self._entry_counter = 0
        self._apply_mode_restrictions()

    def _apply_mode_restrictions(self) -> None:
        """Apply mode-based restrictions to the permission matrix."""
        if self._mode == "workspace_only":
            # Remove external access permissions for non-supervisor roles
            restricted_roles = {"Worker", "Executor", "Researcher", "Critic", "Verifier"}
            for role in restricted_roles:
                perms = self._matrix.get(role, set())
                perms.discard("api_call")
                perms.discard("web_search")
                perms.discard("code_execute")
        elif self._mode == "full_access":
            pass  # No additional restrictions

    # ── 权限矩阵管理 ────────────────────────────────────────────────

    def get_permissions(self, role: str) -> set[str]:
        """获取某角色的权限集合。"""
        return set(self._matrix.get(role, set()))

    def set_permissions(self, role: str, actions: set[str]):
        """设置某角色的权限。"""
        self._matrix[role] = set(actions)

    def add_permission(self, role: str, action: str):
        """给某角色添加权限。"""
        if role not in self._matrix:
            self._matrix[role] = set()
        self._matrix[role].add(action)

    def remove_permission(self, role: str, action: str):
        """移除某角色的权限。"""
        if role in self._matrix:
            self._matrix[role].discard(action)

    def list_roles(self) -> list[str]:
        """列出所有已配置的角色。"""
        return list(self._matrix.keys())

    # ── 操作分级 ────────────────────────────────────────────────────

    def get_action_level(self, action: str) -> ActionLevel:
        """获取操作的风险等级。"""
        return ACTION_LEVELS.get(action, ActionLevel.CRITICAL)

    def is_safe_action(self, action: str) -> bool:
        """检查是否为安全操作（自动执行）。"""
        return self.get_action_level(action) == ActionLevel.SAFE

    def is_risky_action(self, action: str) -> bool:
        """检查是否为风险操作（需 Supervisor 审批）。"""
        return self.get_action_level(action) == ActionLevel.RISKY

    def is_critical_action(self, action: str) -> bool:
        """检查是否为关键操作（需人工确认）。"""
        return self.get_action_level(action) == ActionLevel.CRITICAL

    # ── 权限检查 ────────────────────────────────────────────────────

    def check(
        self,
        role: str,
        action: str,
        task_id: str = "",
        approved_by_supervisor: bool = False,
        approved_by_human: bool = False,
    ) -> PermissionCheckResult:
        """
        检查某角色是否有权执行某操作。

        Args:
            role: 角色名称
            action: 操作名称
            task_id: 任务 ID（用于审计）
            approved_by_supervisor: Supervisor 是否已审批
            approved_by_human: 人工是否已确认
        """
        perms = self.get_permissions(role)
        level = self.get_action_level(action)

        result = PermissionCheckResult(
            role=role,
            action=action,
            allowed=False,
            level=level,
        )

        # 第一步：检查角色是否有该操作的权限
        if action not in perms:
            result.reason = f"Role '{role}' does not have permission for '{action}'"
            self._log_audit(result, task_id, "auto")
            return result

        # 第二步：根据操作等级检查审批
        if level == ActionLevel.SAFE:
            result.allowed = True
            result.requires_approval = "none"
            result.reason = "Safe action, auto-approved"

        elif level == ActionLevel.RISKY:
            result.requires_approval = "supervisor"
            if approved_by_supervisor:
                result.allowed = True
                result.reason = "Risky action approved by Supervisor"
            else:
                result.reason = "Risky action requires Supervisor approval"

        elif level == ActionLevel.CRITICAL:
            result.requires_approval = "human"
            if approved_by_human:
                result.allowed = True
                result.reason = "Critical action approved by human"
            elif approved_by_supervisor:
                result.reason = "Critical action requires human confirmation (Supervisor approval insufficient)"
            else:
                result.reason = "Critical action requires human confirmation"

        self._log_audit(result, task_id, self._get_approval_source(result))
        return result

    def _get_approval_source(self, result: PermissionCheckResult) -> str:
        if result.requires_approval == "none":
            return "auto"
        elif result.allowed and result.requires_approval == "supervisor":
            return "supervisor"
        elif result.allowed and result.requires_approval == "human":
            return "human"
        return "auto"

    # ── 批量检查 ────────────────────────────────────────────────────

    def check_all(
        self,
        role: str,
        actions: list[str],
        task_id: str = "",
        approved_by_supervisor: bool = False,
        approved_by_human: bool = False,
    ) -> list[PermissionCheckResult]:
        """批量检查多个操作的权限。"""
        return [
            self.check(role, a, task_id, approved_by_supervisor, approved_by_human)
            for a in actions
        ]

    def can_role_perform_any(self, role: str, actions: list[str]) -> bool:
        """检查角色是否有权限执行列表中任一操作。"""
        perms = self.get_permissions(role)
        return bool(set(actions) & perms)

    # ── 审计日志 ────────────────────────────────────────────────────

    def _log_audit(self, result: PermissionCheckResult, task_id: str, approval_by: str):
        """记录审计日志。"""
        self._entry_counter += 1
        entry = AuditEntry(
            entry_id=f"AUD-{self._entry_counter:04d}",
            role=result.role,
            action=result.action,
            task_id=task_id,
            allowed=result.allowed,
            reason=result.reason,
            approval_by=approval_by,
        )
        self._audit_log.append(entry)

    def get_audit_log(self, task_id: Optional[str] = None) -> list[AuditEntry]:
        """获取审计日志。"""
        if task_id:
            return [e for e in self._audit_log if e.task_id == task_id]
        return list(self._audit_log)

    def audit_summary(self, task_id: Optional[str] = None) -> dict:
        """返回审计摘要。"""
        entries = self.get_audit_log(task_id)
        total = len(entries)
        allowed = sum(1 for e in entries if e.allowed)
        denied = total - allowed

        by_approval = {"auto": 0, "supervisor": 0, "human": 0}
        for e in entries:
            by_approval[e.approval_by] = by_approval.get(e.approval_by, 0) + 1

        return {
            "total_checks": total,
            "allowed": allowed,
            "denied": denied,
            "by_approval": by_approval,
            "task_id": task_id or "all",
        }
