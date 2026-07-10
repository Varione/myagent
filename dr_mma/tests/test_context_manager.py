"""ContextManager unit tests."""

import pytest
from dr_mma.engine.context_manager import (
    ContextManager,
    RuntimeContext,
    GlobalContext,
    ArtifactContext,
)


class TestRuntimeContext:
    def test_to_dict(self):
        rc = RuntimeContext(task_id="T1", objective="Do X")
        d = rc.to_dict()
        assert d["task_id"] == "T1"
        assert d["objective"] == "Do X"

    def test_token_estimate_basic(self):
        rc = RuntimeContext(task_id="T1", objective="Hello World")
        assert rc.estimated_token_count > 0

    def test_token_estimate_includes_dependencies(self):
        rc = RuntimeContext(
            task_id="T1",
            objective="X",
            direct_dependency_results=["A" * 80, "B" * 80],
        )
        assert rc.estimated_token_count >= 40


class TestGlobalContext:
    def test_to_dict(self):
        gc = GlobalContext(task_id="WF1")
        d = gc.to_dict()
        assert d["task_id"] == "WF1"
        assert d["budget_status"] == "within_budget"

    def test_defaults(self):
        gc = GlobalContext(task_id="W")
        assert gc.key_risks == []
        assert gc.event_log_summary == []


class TestArtifactContext:
    def test_to_dict(self):
        ac = ArtifactContext(artifact_id="A1", summary="Report v1", version=2)
        d = ac.to_dict()
        assert d["artifact_id"] == "A1"
        assert d["version"] == 2

    def test_reference_string(self):
        ac = ArtifactContext(artifact_id="R1", summary="Short summary", version=3)
        ref = ac.reference_string
        assert "R1" in ref
        assert "v3" in ref
        assert "Short summary" in ref

    def test_reference_string_truncates_long_summary(self):
        long = "A" * 200
        ac = ArtifactContext(artifact_id="L1", summary=long, version=1)
        assert len(ac.reference_string.split(": ")[1]) <= 100


class TestContextManagerRuntime:
    def _cm(self, max_tokens=40_000):
        return ContextManager(max_runtime_tokens=max_tokens)

    def test_build_and_get_runtime_context(self):
        cm = self._cm()
        rc = cm.build_runtime_context("T1", "Objective", dependencies=["D1"])
        got = cm.get_runtime_context("T1")
        assert got is not None
        assert got.objective == "Objective"
        assert got.direct_dependency_results == ["D1"]

    def test_get_nonexistent_runtime(self):
        cm = self._cm()
        assert cm.get_runtime_context("NOPE") is None

    def test_token_budget_within(self):
        cm = self._cm(max_tokens=1000)
        cm.build_runtime_context("T1", "Short")
        assert cm.check_token_budget("T1") is True

    def test_token_budget_exceeded(self):
        cm = self._cm(max_tokens=5)
        cm.build_runtime_context("T1", "A" * 100)
        assert cm.check_token_budget("T1") is False

    def test_check_budget_unknown_task_returns_true(self):
        cm = self._cm()
        assert cm.check_token_budget("UNKNOWN") is True

    def test_trim_removes_blackboard_first(self):
        cm = self._cm(max_tokens=5)
        cm.build_runtime_context(
            "T1", "X",
            dependencies=["dep_A" * 20],
            blackboard_entries=["bb1" * 20, "bb2" * 20],
        )
        ctx = cm.trim_runtime_context("T1")
        assert len(ctx.relevant_blackboard_entries) < 2

    def test_trim_keeps_at_least_one_dependency(self):
        cm = self._cm(max_tokens=3)
        cm.build_runtime_context(
            "T1", "X",
            dependencies=["A" * 10, "B" * 10, "C" * 10],
        )
        ctx = cm.trim_runtime_context("T1")
        assert len(ctx.direct_dependency_results) >= 1

    def test_trim_returns_ctx_when_within_budget(self):
        cm = self._cm(max_tokens=10000)
        rc = cm.build_runtime_context("T1", "Short")
        ctx = cm.trim_runtime_context("T1")
        assert ctx is rc

    def test_trim_unknown_task(self):
        cm = self._cm()
        ctx = cm.trim_runtime_context("NOPE")
        assert ctx.task_id == "NOPE"


class TestContextManagerGlobal:
    def _cm(self):
        return ContextManager()

    def test_init_and_get_global(self):
        cm = self._cm()
        gc = cm.init_global_context("WF1")
        assert cm.get_global_context("WF1") is gc

    def test_get_nonexistent_global(self):
        cm = self._cm()
        assert cm.get_global_context("NOPE") is None

    def test_update_subtask_status(self):
        cm = self._cm()
        cm.init_global_context("WF1")
        cm.update_subtask_status("WF1", "S1", "completed")
        gc = cm.get_global_context("WF1")
        assert gc.all_subtask_statuses["S1"] == "completed"

    def test_update_subtask_noop_on_unknown(self):
        cm = self._cm()
        cm.update_subtask_status("NOPE", "S1", "done")
        assert cm.get_global_context("NOPE") is None

    def test_add_event_summary(self):
        cm = self._cm()
        cm.init_global_context("WF1")
        cm.add_event_summary("WF1", {"type": "test"})
        gc = cm.get_global_context("WF1")
        assert len(gc.event_log_summary) == 1

    def test_add_decision_summary(self):
        cm = self._cm()
        cm.init_global_context("WF1")
        cm.add_decision_summary("WF1", {"type": "merge"})
        gc = cm.get_global_context("WF1")
        assert len(gc.decision_log_summary) == 1

    def test_add_key_risk_deduplicates(self):
        cm = self._cm()
        cm.init_global_context("WF1")
        cm.add_key_risk("WF1", "Risk A")
        cm.add_key_risk("WF1", "Risk A")
        cm.add_key_risk("WF1", "Risk B")
        gc = cm.get_global_context("WF1")
        assert len(gc.key_risks) == 2


class TestContextManagerArtifacts:
    def _cm(self):
        return ContextManager()

    def test_register_and_get(self):
        cm = self._cm()
        art = cm.register_artifact("A1", summary="Report", version=2)
        got = cm.get_artifact("A1")
        assert got is not None
        assert got.version == 2

    def test_get_nonexistent_artifact(self):
        cm = self._cm()
        assert cm.get_artifact("NOPE") is None

    def test_list_artifacts(self):
        cm = self._cm()
        cm.register_artifact("A1", summary="X")
        cm.register_artifact("A2", summary="Y")
        assert len(cm.list_artifacts()) == 2

    def test_get_artifact_references(self):
        cm = self._cm()
        cm.register_artifact("A1", summary="First", version=1)
        cm.register_artifact("A2", summary="Second", version=3)
        refs = cm.get_artifact_references(["A1", "A2"])
        assert len(refs) == 2
        assert "A1" in refs[0]
        assert "A2" in refs[1]

    def test_get_artifact_references_skips_missing(self):
        cm = self._cm()
        cm.register_artifact("A1", summary="X")
        refs = cm.get_artifact_references(["A1", "MISSING"])
        assert len(refs) == 1


class TestContextSummary:
    def test_summary_defaults(self):
        cm = ContextManager()
        s = cm.context_summary("T1")
        assert s["task_id"] == "T1"
        assert s["has_runtime_context"] is False
        assert s["has_global_context"] is False
        assert s["artifact_count"] == 0

    def test_summary_with_data(self):
        cm = ContextManager()
        cm.build_runtime_context("T1", "Obj")
        cm.init_global_context("T1")
        cm.update_subtask_status("T1", "S1", "done")
        cm.register_artifact("A1", summary="X")
        s = cm.context_summary("T1")
        assert s["has_runtime_context"] is True
        assert s["has_global_context"] is True
        assert s["subtask_count"] == 1
        assert s["artifact_count"] == 1
