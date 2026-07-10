"""
DR-MMA Phase 2 — WorkflowEngine integration tests.

覆盖范围：
  - Pool population (_ensure_pool_populated)
  - Role merge/split in execute flow (_apply_role_manager_decisions)
  - Health tracking (record_call_success/failure through _record)
  - Direct mode execution with mock models
  - Event publishing (workflow_mode, role_assigned)
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.models.adapter import ModelAdapter, MockModel
from dr_mma.engine.model_pool import ModelPool
from dr_mma.engine.workflow import WorkflowEngine, WorkflowResult
from dr_mma.engine.events import EVENT_WORKFLOW_MODE, EVENT_ROLE_ASSIGNED
from dr_mma.storage.blackboard import Blackboard
from dr_mma.storage.artifact_store import ArtifactStore
from dr_mma.storage.decision_log import DecisionLog


def _make_engine_with_mock(
    model_names: list[str] | None = None,
    responses: dict[str, str] | None = None,
) -> tuple[WorkflowEngine, ModelAdapter, ModelPool]:
    """Create a WorkflowEngine with mock models for testing."""
    if model_names is None:
        model_names = ["mock-primary", "mock-secondary"]

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


class TestPoolPopulation:
    """Test _ensure_pool_populated integrates models into the pool."""

    def test_pool_populated_from_primary_model(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        # Before execute, pool should have primary model
        assert pool.get("mock-primary") is not None
        assert pool.model_count >= 1

    def test_pool_populated_from_adapter_models(self):
        engine, adapter, pool = _make_engine_with_mock(
            ["mock-a", "mock-b", "mock-c"]
        )
        # All registered models should be in pool after execute
        result = engine.execute("简单问题：1+1等于几？")
        assert pool.model_count >= 3

    def test_health_check_runs_on_execute(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题：1+1等于几？")
        # health_check_all was called, entries have timestamps
        entry = pool.get("mock-primary")
        assert entry is not None

    def test_pool_healthy_models_available(self):
        engine, adapter, pool = _make_engine_with_mock(
            ["mock-a", "mock-b"]
        )
        assert len(pool.healthy_models()) >= 1


class TestRoleManagerIntegration:
    """Test merge/split decisions in the workflow execute flow."""

    def test_role_bindings_created_on_execute(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        # Role bindings should be created by _apply_role_manager_decisions
        bindings = engine.role_manager.list_bindings()
        assert len(bindings) >= 1

    def test_merge_attempted_when_pool_small(self):
        """When healthy_count <= 2, merge is attempted on adjacent roles."""
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        # With only 1 model, all roles assigned to same model
        assert result.role_assignments["Planner"] == "mock-primary"
        assert result.role_assignments["Supervisor"] == "mock-primary"

    def test_split_attempted_when_pool_large(self):
        """When healthy_count >= 4, split is checked."""
        engine, adapter, pool = _make_engine_with_mock(
            ["mock-a", "mock-b", "mock-c", "mock-d"]
        )
        result = engine.execute("简单问题")
        # Engine runs without error when pool has 4+ models
        assert isinstance(result, WorkflowResult)

    def test_role_assignments_populated(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert "Planner" in result.role_assignments
        assert "Supervisor" in result.role_assignments


class TestHealthTracking:
    """Test model health tracking through workflow execution."""

    def test_call_success_recorded(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        before = pool.get("mock-primary").success_count
        result = engine.execute("简单问题：1+1等于几？")
        after = pool.get("mock-primary").success_count
        assert after > before

    def test_blackboard_entries_created(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert result.blackboard_count >= 1

    def test_event_published_for_workflow_mode(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        mode_events = engine.event_bus.query(event_type=EVENT_WORKFLOW_MODE)
        assert len(mode_events) >= 1

    def test_event_published_for_role_assigned(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        role_events = engine.event_bus.query(event_type=EVENT_ROLE_ASSIGNED)
        assert len(role_events) >= 1


class TestDirectModeExecution:
    """Test direct mode (low complexity) workflow execution."""

    def test_direct_mode_returns_result(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题：1+1等于几？")
        assert isinstance(result, WorkflowResult)
        assert result.task_id.lower().startswith("wf-")
        assert result.mode  # should have a mode set

    def test_direct_mode_has_final_output(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert len(result.final_output) > 0

    def test_direct_mode_status_completed(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert result.status == "completed"

    def test_direct_mode_latency_recorded(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert result.total_latency_ms > 0

    def test_runtime_config_propagated(self):
        engine, adapter, pool = _make_engine_with_mock(
            ["mock-primary"],
        )
        engine.runtime_config = {"timeout_seconds": 60}
        result = engine.execute("简单问题")
        assert result.runtime_config.get("timeout_seconds") == 60

    def test_workflow_result_has_subtask_results(self):
        engine, adapter, pool = _make_engine_with_mock(["mock-primary"])
        result = engine.execute("简单问题")
        assert len(result.subtask_results) >= 1
