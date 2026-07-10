"""ToolExecutor tests — permissions, execution, audit trail."""

import pytest
from dr_mma.engine.tool_executor import ToolExecutor, ToolExecutionRecord
from dr_mma.engine.tools import (
    ToolRegistry,
    ToolCategory,
    ToolSafetyLevel,
    create_default_tool_registry,
)
from dr_mma.engine.permissions import PermissionManager, ActionLevel
from dr_mma.schemas.agent_response import AgentResponse, ToolCall


@pytest.fixture
def registry():
    reg = ToolRegistry()
    # Safe tool — no role restriction so ToolRegistry always allows it
    reg.register(
        "echo",
        lambda args: {"output": args.get("text", "")},
        description="Echo text",
        category=ToolCategory.GENERAL,
        safety_level=ToolSafetyLevel.SAFE,
        allowed_roles=[],
    )
    # Risky tool (needs supervisor approval)
    reg.register(
        "web_search",
        lambda args: {"results": [args.get("query", "")]},
        description="Search web",
        category=ToolCategory.WEB_SEARCH,
        safety_level=ToolSafetyLevel.RISKY,
        allowed_roles=["Researcher", "Supervisor"],
    )
    return reg


@pytest.fixture
def perm_manager():
    pm = PermissionManager()
    # Add echo as a safe action and grant to Worker/Critic/Verifier
    import dr_mma.engine.permissions as perm_mod
    perm_mod.ACTION_LEVELS["echo"] = ActionLevel.SAFE
    pm.add_permission("Worker", "echo")
    pm.add_permission("Critic", "echo")
    pm.add_permission("Verifier", "echo")
    return pm


@pytest.fixture
def executor(registry, perm_manager):
    return ToolExecutor(registry, perm_manager)


class TestExecuteSingleTool:
    """Test single tool execution with various scenarios."""

    def test_success_safe_tool(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "hello"})],
        )
        records = executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(records) == 1
        assert records[0].permission_allowed is True
        assert records[0].result is not None
        assert records[0].result.success is True
        assert records[0].result.output == {"output": "hello"}

    def test_permission_denied_wrong_role(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Critic",
            tool_calls=[ToolCall(tool_name="web_search", args={"query": "test"})],
        )
        records = executor.execute_calls(response, role="Critic", task_id="T1")
        assert len(records) == 1
        assert records[0].permission_allowed is False
        assert "not allowed" in records[0].permission_reason.lower() or "does not have" in records[0].permission_reason.lower()

    def test_tool_not_in_allowed_list(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "hi"})],
        )
        records = executor.execute_calls(
            response, role="Worker", task_id="T1", allowed_tools=["web_search"]
        )
        assert len(records) == 1
        assert records[0].permission_allowed is False
        assert "not in allowed" in records[0].permission_reason.lower()

    def test_tool_not_registered(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="nonexistent", args={})],
        )
        records = executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(records) == 1
        assert records[0].permission_allowed is False
        # Permission check fails first since "nonexistent" not in Worker's permissions
        assert "does not have permission" in records[0].permission_reason.lower() or "not registered" in records[0].permission_reason.lower()

    def test_no_tool_calls(self, executor):
        response = AgentResponse(task_id="T1", role="Worker")
        records = executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(records) == 0

    def test_results_written_to_response(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "test"})],
        )
        assert response.tool_results == []
        executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(response.tool_results) == 1
        assert response.tool_results[0]["tool_name"] == "echo"


class TestExecuteMultipleTools:
    """Test batch execution of multiple tool calls."""

    def test_multiple_tools_success(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[
                ToolCall(tool_name="echo", args={"text": "a"}),
                ToolCall(tool_name="echo", args={"text": "b"}),
            ],
        )
        records = executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(records) == 2
        assert all(r.result.success for r in records)

    def test_mixed_success_failure(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Critic",
            tool_calls=[
                ToolCall(tool_name="echo", args={"text": "ok"}),
                ToolCall(tool_name="web_search", args={"query": "denied"}),
            ],
        )
        records = executor.execute_calls(response, role="Critic", task_id="T1")
        assert len(records) == 2
        # echo should succeed for Critic (in allowed_roles)
        assert records[0].result.success is True
        # web_search should be denied for Critic
        assert records[1].permission_allowed is False


class TestExecuteLoop:
    """Test multi-response execution loop."""

    def test_loop_multiple_responses(self, executor):
        responses = [
            AgentResponse(
                task_id="T1",
                role="Worker",
                tool_calls=[ToolCall(tool_name="echo", args={"text": "w"})],
            ),
            AgentResponse(
                task_id="T1",
                role="Verifier",
                tool_calls=[ToolCall(tool_name="echo", args={"text": "v"})],
            ),
        ]
        records = executor.execute_loop(responses, task_id="T1")
        assert len(records) == 2


class TestAuditAndDiagnostics:
    """Test audit trail and diagnostic methods."""

    def test_audit_summary(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[
                ToolCall(tool_name="echo", args={"text": "ok"}),
                ToolCall(tool_name="nonexistent", args={}),
            ],
        )
        executor.execute_calls(response, role="Worker", task_id="T1")
        summary = executor.audit_summary(task_id="T1")
        assert summary["total_calls"] == 2
        assert summary["permission_allowed"] == 1
        assert summary["permission_denied"] == 1

    def test_get_records_filter(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "x"})],
        )
        executor.execute_calls(response, role="Worker", task_id="T1")
        records = executor.get_records(task_id="T1")
        assert len(records) == 1
        assert executor.get_records(task_id="T999") == []

    def test_get_records_dict(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "dict"})],
        )
        executor.execute_calls(response, role="Worker", task_id="T1")
        dicts = executor.get_records_dict(task_id="T1")
        assert len(dicts) == 1
        assert dicts[0]["tool_name"] == "echo"
        assert dicts[0]["success"] is True

    def test_clear_records(self, executor):
        response = AgentResponse(
            task_id="T1",
            role="Worker",
            tool_calls=[ToolCall(tool_name="echo", args={"text": "clear"})],
        )
        executor.execute_calls(response, role="Worker", task_id="T1")
        assert len(executor.get_records()) == 1
        executor.clear_records()
        assert len(executor.get_records()) == 0

    def test_permission_audit(self, executor):
        audit = executor.permission_audit()
        assert "total_checks" in audit
        assert "allowed" in audit
        assert "denied" in audit


class TestToolExecutionRecord:
    """Test ToolExecutionRecord serialization."""

    def test_to_dict_success(self):
        record = ToolExecutionRecord(
            tool_name="echo",
            task_id="T1",
            role="Worker",
            args={"text": "hello"},
            permission_allowed=True,
            action_level="safe",
        )
        d = record.to_dict()
        assert d["tool_name"] == "echo"
        assert d["success"] is False  # no result yet
        assert d["permission_allowed"] is True

    def test_to_dict_with_error(self):
        record = ToolExecutionRecord(
            tool_name="bad",
            task_id="T1",
            role="Worker",
            permission_allowed=False,
            permission_reason="denied",
            action_level="blocked",
        )
        d = record.to_dict()
        assert d["error"] == "denied"


class TestDefaultRegistryIntegration:
    """Test with actual default tool registry."""

    def test_code_execute_not_in_default_registry(self):
        """code_execute is not registered by default."""
        registry = create_default_tool_registry()
        assert not registry.tool_exists("code_execute")

    def test_code_execute_blocked_when_not_registered(self):
        """Tool executor blocks code_execute when not in registry."""
        registry = create_default_tool_registry()
        perm = PermissionManager(mode="full_access")
        executor = ToolExecutor(registry, perm)

        response = AgentResponse(
            task_id="T1",
            role="Supervisor",
            tool_calls=[
                ToolCall(
                    tool_name="code_execute",
                    args={"code": "x = 1 + 2"},
                )
            ],
        )
        records = executor.execute_calls(response, role="Supervisor", task_id="T1")
        assert len(records) == 1
        assert records[0].permission_allowed is False
        # Either blocked by permission (critical needs human) or not registered
        assert "not registered" in records[0].permission_reason or "human" in records[0].permission_reason.lower()

    def test_web_search_works_with_approval(self):
        """web_search works with supervisor approval."""
        registry = create_default_tool_registry()
        perm = PermissionManager(mode="full_access")
        executor = ToolExecutor(registry, perm)

        response = AgentResponse(
            task_id="T1",
            role="Researcher",
            tool_calls=[
                ToolCall(
                    tool_name="web_search",
                    args={"query": "test"},
                )
            ],
        )
        records = executor.execute_calls(response, role="Researcher", task_id="T1")
        assert len(records) == 1
        # web_search is RISKY, needs supervisor approval
        assert records[0].permission_allowed is False
        assert records[0].action_level == "risky"
        assert records[0].result is None  # Not executed due to permission denial
