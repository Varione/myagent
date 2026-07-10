"""BudgetController unit tests."""

import pytest
from dr_mma.engine.budget_controller import (
    BudgetController,
    BudgetConfig,
    BudgetUsage,
)


class TestBudgetInitialization:
    def test_initialize_returns_config(self):
        bc = BudgetController()
        cfg = bc.initialize("T1", max_model_calls=20)
        assert isinstance(cfg, BudgetConfig)
        assert cfg.task_id == "T1"
        assert cfg.max_model_calls == 20

    def test_initialize_defaults(self):
        bc = BudgetController()
        cfg = bc.initialize("T1")
        assert cfg.max_model_calls == 30
        assert cfg.max_tool_calls == 10
        assert cfg.max_total_tokens == 120_000

    def test_get_config(self):
        bc = BudgetController()
        bc.initialize("T1")
        assert bc.get_config("T1") is not None
        assert bc.get_config("NOPE") is None

    def test_usage_created_on_init(self):
        bc = BudgetController()
        bc.initialize("T1")
        usage = bc.get_usage("T1")
        assert usage is not None
        assert usage.status == "within_budget"


class TestModelCallRecording:
    def _setup(self, **kwargs):
        bc = BudgetController()
        bc.initialize("T1", **kwargs)
        return bc

    def test_record_model_call_increments(self):
        bc = self._setup()
        bc.record_model_call("T1", tokens=4000, cost=0.05)
        usage = bc.get_usage("T1")
        assert usage.model_calls_used == 1
        assert usage.tokens_consumed == 4000
        assert usage.estimated_cost == pytest.approx(0.05)

    def test_record_high_cost_call(self):
        bc = self._setup()
        bc.record_model_call("T1", is_high_cost=True)
        usage = bc.get_usage("T1")
        assert usage.high_cost_calls == 1

    def test_record_unknown_task_noop(self):
        bc = BudgetController()
        usage = bc.record_model_call("NOPE", tokens=100)
        assert usage.task_id == "NOPE"


class TestToolCallRecording:
    def _setup(self):
        bc = BudgetController()
        bc.initialize("T1")
        return bc

    def test_record_tool_call_increments(self):
        bc = self._setup()
        bc.record_tool_call("T1")
        usage = bc.get_usage("T1")
        assert usage.tool_calls_used == 1


class TestDebateRecording:
    def _setup(self):
        bc = BudgetController()
        bc.initialize("T1", max_debate_rounds=2)
        return bc

    def test_record_debate_round_increments(self):
        bc = self._setup()
        bc.record_debate_round("T1")
        usage = bc.get_usage("T1")
        assert usage.debate_rounds_used == 1


class TestRetryRecording:
    def _setup(self):
        bc = BudgetController()
        bc.initialize("T1", max_retries_per_node=2)
        return bc

    def test_record_retry_tracks_per_node(self):
        bc = self._setup()
        bc.record_retry("T1", "N1")
        bc.record_retry("T1", "N1")
        usage = bc.get_usage("T1")
        assert usage.retries_used["N1"] == 2

    def test_record_retry_different_nodes(self):
        bc = self._setup()
        bc.record_retry("T1", "N1")
        bc.record_retry("T1", "N2")
        usage = bc.get_usage("T1")
        assert usage.retries_used["N1"] == 1
        assert usage.retries_used["N2"] == 1


class TestStatusTransitions:
    def test_within_budget(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_model_calls=100)
        bc.record_model_call("T1")
        assert bc.get_usage("T1").status == "within_budget"

    def test_reaches_warning(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_model_calls=10)
        for _ in range(9):
            bc.record_model_call("T1")
        assert bc.get_usage("T1").status == "warning"

    def test_reaches_exceeded(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_model_calls=5)
        for _ in range(5):
            bc.record_model_call("T1")
        assert bc.get_usage("T1").status == "exceeded"

    def test_warning_from_tool_calls(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_tool_calls=5, max_model_calls=100)
        for _ in range(4):
            bc.record_tool_call("T1")
        assert bc.get_usage("T1").status == "warning"

    def test_exceeded_from_tokens(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_total_tokens=100, max_model_calls=1000)
        bc.record_tokens("T1", 100)
        assert bc.get_usage("T1").status == "exceeded"

    def test_exceeded_from_debate(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_debate_rounds=2, max_model_calls=1000)
        bc.record_debate_round("T1")
        bc.record_debate_round("T1")
        assert bc.get_usage("T1").status == "exceeded"

    def test_exceeded_from_retry(self):
        bc = BudgetController(warning_threshold=0.8)
        bc.initialize("T1", max_retries_per_node=1, max_model_calls=1000)
        bc.record_retry("T1", "N1")
        assert bc.get_usage("T1").status == "exceeded"

    def test_warning_recorded_on_transition(self):
        bc = BudgetController(warning_threshold=0.5)
        bc.initialize("T1", max_model_calls=10)
        for _ in range(6):
            bc.record_model_call("T1")
        usage = bc.get_usage("T1")
        assert len(usage.warnings_issued) >= 1


class TestCanCheckMethods:
    def test_can_call_model_true(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=10)
        assert bc.can_call_model("T1") is True

    def test_can_call_model_false(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=2)
        bc.record_model_call("T1")
        bc.record_model_call("T1")
        assert bc.can_call_model("T1") is False

    def test_can_call_tool_true(self):
        bc = BudgetController()
        bc.initialize("T1", max_tool_calls=5)
        assert bc.can_call_tool("T1") is True

    def test_can_call_tool_false(self):
        bc = BudgetController()
        bc.initialize("T1", max_tool_calls=1)
        bc.record_tool_call("T1")
        assert bc.can_call_tool("T1") is False

    def test_can_debate_true(self):
        bc = BudgetController()
        bc.initialize("T1", max_debate_rounds=3)
        assert bc.can_debate("T1") is True

    def test_can_debate_false(self):
        bc = BudgetController()
        bc.initialize("T1", max_debate_rounds=1)
        bc.record_debate_round("T1")
        assert bc.can_debate("T1") is False

    def test_can_retry_true(self):
        bc = BudgetController()
        bc.initialize("T1", max_retries_per_node=2)
        assert bc.can_retry("T1", "N1") is True

    def test_can_retry_false(self):
        bc = BudgetController()
        bc.initialize("T1", max_retries_per_node=1)
        bc.record_retry("T1", "N1")
        assert bc.can_retry("T1", "N1") is False

    def test_unknown_task_returns_true(self):
        bc = BudgetController()
        assert bc.is_within_budget("NOPE") is True
        assert bc.can_call_model("NOPE") is True


class TestBudgetAdjustment:
    def test_increase_model_calls(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=5)
        for _ in range(5):
            bc.record_model_call("T1")
        assert bc.can_call_model("T1") is False

        bc.increase_budget("T1", extra_model_calls=5)
        assert bc.can_call_model("T1") is True

    def test_increase_multiple_dimensions(self):
        bc = BudgetController()
        bc.initialize("T1", max_tool_calls=2, max_debate_rounds=1)
        bc.record_tool_call("T1")
        bc.record_tool_call("T1")
        assert bc.can_call_tool("T1") is False

        bc.increase_budget("T1", extra_tool_calls=3, extra_debate_rounds=2)
        cfg = bc.get_config("T1")
        assert cfg.max_tool_calls == 5
        assert cfg.max_debate_rounds == 3

    def test_increase_unknown_task_noop(self):
        bc = BudgetController()
        result = bc.increase_budget("NOPE", extra_model_calls=10)
        assert result is None

    def test_reset_usage(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=10)
        for _ in range(5):
            bc.record_model_call("T1", tokens=1000, cost=0.1)
        bc.reset_usage("T1")
        usage = bc.get_usage("T1")
        assert usage.model_calls_used == 0
        assert usage.tokens_consumed == 0
        assert usage.status == "within_budget"

    def test_reset_unknown_task_noop(self):
        bc = BudgetController()
        assert bc.reset_usage("NOPE") is None


class TestBudgetSummary:
    def test_summary_not_initialized(self):
        bc = BudgetController()
        s = bc.budget_summary("T1")
        assert "error" in s

    def test_summary_with_data(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=20, max_tool_calls=5)
        bc.record_model_call("T1", tokens=4000, cost=0.05)
        bc.record_tool_call("T1")
        s = bc.budget_summary("T1")
        assert s["model_calls"] == "1/20"
        assert s["tool_calls"] == "1/5"
        assert s["status"] == "within_budget"

    def test_summary_format(self):
        bc = BudgetController()
        bc.initialize("T1", max_model_calls=30, max_total_tokens=120_000)
        s = bc.budget_summary("T1")
        assert "tokens" in s
        assert "estimated_cost" in s
        assert "warnings_count" in s


class TestDataclasses:
    def test_config_to_dict(self):
        cfg = BudgetConfig(task_id="T1", max_model_calls=25)
        d = cfg.to_dict()
        assert d["task_id"] == "T1"
        assert d["max_model_calls"] == 25

    def test_usage_to_dict(self):
        u = BudgetUsage(task_id="T1", model_calls_used=3, estimated_cost=0.12345)
        d = u.to_dict()
        assert d["model_calls_used"] == 3
        assert d["estimated_cost"] == pytest.approx(0.1235, rel=1e-3)
