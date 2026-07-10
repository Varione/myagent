"""Permissions unit tests."""

import pytest
from dr_mma.engine.permissions import (
    PermissionManager,
    ActionLevel,
    PermissionCheckResult,
    AuditEntry,
)


class TestPermissionMatrix:
    def test_default_roles_exist(self):
        pm = PermissionManager()
        roles = pm.list_roles()
        assert "Supervisor" in roles
        assert "Executor" in roles
        assert "Critic" in roles

    def test_get_permissions(self):
        pm = PermissionManager()
        perms = pm.get_permissions("Planner")
        assert "blackboard_read" in perms
        assert "blackboard_write" in perms

    def test_supervisor_has_most_permissions(self):
        pm = PermissionManager()
        sup_perms = pm.get_permissions("Supervisor")
        worker_perms = pm.get_permissions("Executor")
        assert len(sup_perms) >= len(worker_perms)

    def test_set_permissions(self):
        pm = PermissionManager()
        pm.set_permissions("CustomRole", {"action_a", "action_b"})
        perms = pm.get_permissions("CustomRole")
        assert perms == {"action_a", "action_b"}

    def test_add_permission(self):
        pm = PermissionManager()
        pm.add_permission("Critic", "custom_action")
        assert "custom_action" in pm.get_permissions("Critic")

    def test_remove_permission(self):
        pm = PermissionManager()
        pm.remove_permission("Planner", "blackboard_write")
        assert "blackboard_write" not in pm.get_permissions("Planner")

    def test_unknown_role_empty_permissions(self):
        pm = PermissionManager()
        assert pm.get_permissions("UnknownRole") == set()


class TestActionLevels:
    def test_safe_actions(self):
        pm = PermissionManager()
        assert pm.is_safe_action("blackboard_read") is True
        assert pm.is_safe_action("database_query") is True

    def test_code_execute_is_critical(self):
        """code_execute is CRITICAL and requires human approval."""
        pm = PermissionManager()
        assert pm.is_critical_action("code_execute") is True

    def test_risky_actions(self):
        pm = PermissionManager()
        assert pm.is_risky_action("api_call") is True
        assert pm.is_risky_action("web_search") is True

    def test_critical_actions(self):
        pm = PermissionManager()
        assert pm.is_critical_action("delete_file") is True
        assert pm.is_critical_action("send_external_message") is True

    def test_unknown_action_defaults_to_critical(self):
        pm = PermissionManager()
        assert pm.get_action_level("unknown_xyz") == ActionLevel.CRITICAL


class TestPermissionCheck:
    def _setup(self):
        return PermissionManager(mode="full_access")

    def test_safe_action_allowed(self):
        pm = self._setup()
        r = pm.check("Planner", "blackboard_read", task_id="T1")
        assert r.allowed is True
        assert r.requires_approval == "none"

    def test_permission_denied_no_role(self):
        pm = self._setup()
        r = pm.check("Critic", "code_execute", task_id="T1")
        assert r.allowed is False
        assert "does not have permission" in r.reason

    def test_risky_action_requires_supervisor(self):
        pm = self._setup()
        r = pm.check("Executor", "api_call", task_id="T1")
        assert r.allowed is False
        assert r.requires_approval == "supervisor"

    def test_risky_action_approved_by_supervisor(self):
        pm = self._setup()
        r = pm.check("Executor", "api_call", task_id="T1", approved_by_supervisor=True)
        assert r.allowed is True

    def test_critical_action_requires_human(self):
        """Critical 操作即使 Supervisor 有权限也需要人工确认。"""
        pm = self._setup()
        pm.add_permission("Supervisor", "delete_file")
        r = pm.check("Supervisor", "delete_file", task_id="T1")
        assert r.allowed is False
        assert r.requires_approval == "human"

    def test_critical_action_supervisor_approval_insufficient(self):
        pm = self._setup()
        pm.add_permission("Supervisor", "delete_file")
        r = pm.check(
            "Supervisor", "delete_file", task_id="T1", approved_by_supervisor=True
        )
        assert r.allowed is False
        assert "Supervisor approval insufficient" in r.reason

    def test_critical_action_approved_by_human(self):
        pm = self._setup()
        pm.add_permission("Supervisor", "delete_file")
        r = pm.check("Supervisor", "delete_file", task_id="T1", approved_by_human=True)
        assert r.allowed is True

    def test_to_dict(self):
        pm = self._setup()
        r = pm.check("Planner", "blackboard_read")
        d = r.to_dict()
        assert d["role"] == "Planner"
        assert d["allowed"] is True
        assert d["level"] == "safe"


class TestBatchCheck:
    def _setup(self):
        return PermissionManager()

    def test_check_all_returns_list(self):
        pm = self._setup()
        results = pm.check_all("Planner", ["blackboard_read", "blackboard_write"])
        assert len(results) == 2
        assert all(r.allowed for r in results)

    def test_can_role_perform_any(self):
        pm = self._setup()
        assert pm.can_role_perform_any("Planner", ["blackboard_read", "code_execute"]) is True

    def test_can_role_perform_any_none(self):
        pm = self._setup()
        assert pm.can_role_perform_any("Critic", ["delete_file", "modify_system_config"]) is False


class TestAuditLog:
    def _setup(self):
        return PermissionManager()

    def test_audit_entries_created(self):
        pm = self._setup()
        pm.check("Planner", "blackboard_read", task_id="T1")
        log = pm.get_audit_log("T1")
        assert len(log) == 1
        assert log[0].role == "Planner"

    def test_audit_entries_filtered_by_task(self):
        pm = self._setup()
        pm.check("Planner", "blackboard_read", task_id="T1")
        pm.check("Worker", "code_execute", task_id="T2")
        assert len(pm.get_audit_log("T1")) == 1
        assert len(pm.get_audit_log("T2")) == 1
        assert len(pm.get_audit_log()) == 2

    def test_audit_entry_to_dict(self):
        entry = AuditEntry(entry_id="A1", role="W", action="X", allowed=True)
        d = entry.to_dict()
        assert d["entry_id"] == "A1"
        assert d["allowed"] is True

    def test_audit_summary(self):
        pm = self._setup()
        pm.check("Planner", "blackboard_read", task_id="T1")
        pm.check("Critic", "code_execute", task_id="T1")
        s = pm.audit_summary("T1")
        assert s["total_checks"] == 2
        assert s["allowed"] == 1
        assert s["denied"] == 1

    def test_audit_summary_all(self):
        pm = self._setup()
        pm.check("Planner", "blackboard_read", task_id="T1")
        s = pm.audit_summary()
        assert s["task_id"] == "all"


class TestPermissionMatrixDefaults:
    """验证架构计划中定义的权限矩阵。"""

    def _setup(self):
        return PermissionManager(mode="full_access")

    def test_planner_cannot_modify_dag(self):
        pm = self._setup()
        r = pm.check("Planner", "modify_dag")
        assert r.allowed is False

    def test_executor_can_code_execute_full_access(self):
        """In full_access mode, executor can request code_execute (but needs human approval)."""
        pm = self._setup()
        r = pm.check("Executor", "code_execute")
        # code_execute is CRITICAL, so requires human approval even if role has permission
        assert r.requires_approval == "human"

    def test_researcher_can_web_search_full_access(self):
        """In full_access mode, researcher can request web_search (needs supervisor approval)."""
        pm = self._setup()
        r = pm.check("Researcher", "web_search")
        assert r.requires_approval == "supervisor"

    def test_workspace_only_restricts_external_access(self):
        """In workspace_only mode, non-supervisor roles lose external access."""
        pm = PermissionManager(mode="workspace_only")
        r = pm.check("Executor", "code_execute")
        assert r.allowed is False
        r = pm.check("Researcher", "web_search")
        assert r.allowed is False

    def test_verifier_can_database_query(self):
        pm = self._setup()
        r = pm.check("Verifier", "database_query")
        assert r.allowed is True

    def test_supervisor_can_modify_dag(self):
        pm = self._setup()
        r = pm.check("Supervisor", "modify_dag")
        assert r.requires_approval == "supervisor"

    def test_no_role_can_delete_file(self):
        pm = self._setup()
        for role in pm.list_roles():
            perms = pm.get_permissions(role)
            assert "delete_file" not in perms, f"{role} should not have delete_file"
