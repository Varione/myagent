"""Workflow tool call integration tests — end-to-end tool execution in workflow."""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.models.adapter import ModelAdapter, MockModel
from dr_mma.engine.model_pool import ModelPool
from dr_mma.engine.workflow import WorkflowEngine, WorkflowResult
from dr_mma.engine.permissions import PermissionManager
from dr_mma.storage.blackboard import Blackboard
from dr_mma.storage.artifact_store import ArtifactStore
from dr_mma.storage.decision_log import DecisionLog


def _make_engine_with_mock(
    model_names: list[str] | None = None,
    responses: dict[str, str] | None = None,
) -> tuple[WorkflowEngine, ModelAdapter, ModelPool]:
    """Create a WorkflowEngine with mock models for testing."""
    if model_names is None:
        model_names = ["mock-primary"]

    adapter = ModelAdapter()
    for name in model_names:
        mock = MockModel(name, responses=responses)
        adapter.register(name, mock)

    pool = ModelPool()
    for name in model_names:
        pool.register(name, name=name, provider="local")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        bb_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        dl_path = f.name

    blackboard = Blackboard(bb_path)
    artifact_store = ArtifactStore(tempfile.gettempdir())
    decision_log = DecisionLog(dl_path)

    engine = WorkflowEngine(
        adapter=adapter,
        blackboard=blackboard,
        artifact_store=artifact_store,
        decision_log=decision_log,
        main_model=model_names[0],
        model_pool=pool,
    )
    return engine, adapter, pool


class TestWorkflowToolExecutorSetup:
    """Verify WorkflowEngine has ToolExecutor properly initialized."""

    def test_tool_executor_exists(self):
        engine, _, _ = _make_engine_with_mock()
        assert engine.tool_executor is not None
        assert engine.tool_registry is not None
        assert engine.permission_manager is not None

    def test_default_tools_registered(self):
        engine, _, _ = _make_engine_with_mock()
        assert engine.tool_registry.tool_count >= 4
        assert engine.tool_registry.tool_exists("code_execute")
        assert engine.tool_registry.tool_exists("file_parse")

    def test_permission_manager_initialized(self):
        engine, _, _ = _make_engine_with_mock()
        assert isinstance(engine.permission_manager, PermissionManager)
        roles = engine.permission_manager.list_roles()
        assert "Supervisor" in roles


class TestWorkflowDirectModeWithTools:
    """Test direct mode workflow executes tool calls from model response."""

    def test_direct_mode_no_tool_calls(self):
        responses = {
            "直接执行": '{"status": "completed", "summary": "done", "content": "Result: 42"}',
        }
        engine, _, _ = _make_engine_with_mock(responses=responses)
        result = engine.execute("简单问题")
        assert result.status == "completed"
        assert "42" in result.final_output

    def test_direct_mode_with_tool_calls(self):
        responses = {
            "直接执行": (
                '{"status": "completed", "summary": "computed", "content": "Result: 3", '
                '"tool_calls": [{"tool_name": "code_execute", "args": {"code": "print(1+2)"}}]}'
            ),
        }
        engine, _, _ = _make_engine_with_mock(responses=responses)
        result = engine.execute("简单计算")
        assert result.status == "completed"
        audit = engine.tool_executor.audit_summary()
        assert audit["total_calls"] >= 1


class TestWorkflowToolEventPublishing:
    """Test that tool execution events are published to event bus."""

    def test_tool_executed_event_published(self):
        responses = {
            "直接执行": (
                '{"status": "completed", "summary": "done", "content": "ok", '
                '"tool_calls": [{"tool_name": "code_execute", "args": {"code": "1+1"}}]}'
            ),
        }
        engine, _, _ = _make_engine_with_mock(responses=responses)
        result = engine.execute("简单计算")
        events = engine.event_bus.all_events()
        tool_events = [e for e in events if e.event_type == "TOOL_EXECUTED"]
        assert len(tool_events) >= 1
        assert tool_events[0].payload.get("tool_name") == "code_execute"


class TestWorkflowToolAllowedList:
    """Test that allowed_tools from runtime_config filters tool calls."""

    def test_allowed_tools_filters(self):
        responses = {
            "直接执行": (
                '{"status": "completed", "summary": "done", "content": "ok", '
                '"tool_calls": ['
                '  {"tool_name": "code_execute", "args": {"code": "1"}},'
                '  {"tool_name": "web_search", "args": {"query": "x"}}'
                ']}'
            ),
        }
        engine, _, _ = _make_engine_with_mock(responses=responses)
        engine.runtime_config["allowed_tools"] = ["code_execute"]
        result = engine.execute("简单问题")
        assert result.status == "completed"
        records = engine.tool_executor.get_records_dict()
        web_search_records = [r for r in records if r["tool_name"] == "web_search"]
        assert len(web_search_records) >= 1
        assert web_search_records[0]["permission_allowed"] is False


class TestWorkflowPermissionIntegration:
    """Test permission checks are enforced during workflow execution."""

    def test_permission_denied_recorded(self):
        responses = {
            "直接执行": (
                '{"status": "completed", "summary": "tried", "content": "result", '
                '"tool_calls": [{"tool_name": "web_search", "args": {"query": "test"}}]}'
            ),
        }
        engine, _, _ = _make_engine_with_mock(responses=responses)
        result = engine.execute("简单搜索")
        assert result.status == "completed"
        audit = engine.tool_executor.audit_summary()
        assert audit["total_calls"] >= 1
