"""
DR-MMA Phase 2 — ModelPool unit tests.

覆盖范围：
  - 注册 / 注销 / 查询
  - 健康检查状态迁移 (healthy → degraded → unhealthy → unavailable)
  - 成功/失败记录
  - 可靠性分数与最佳模型选择
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.engine.model_pool import ModelPool, ModelEntry


class TestModelPoolRegistration:
    def test_register_and_get(self):
        pool = ModelPool()
        entry = pool.register("m1", name="Model 1", provider="local")
        assert pool.get("m1") is entry
        assert entry.model_id == "m1"
        assert entry.name == "Model 1"
        assert entry.status == "healthy"

    def test_register_defaults(self):
        pool = ModelPool()
        entry = pool.register("m2")
        assert entry.provider == "local"
        assert entry.context_length == 128_000
        assert entry.health_check_interval == 300.0

    def test_unregister(self):
        pool = ModelPool()
        pool.register("m3")
        removed = pool.unregister("m3")
        assert removed is not None
        assert pool.get("m3") is None

    def test_unregister_nonexistent(self):
        pool = ModelPool()
        assert pool.unregister("nope") is None

    def test_list_models_and_counts(self):
        pool = ModelPool()
        pool.register("a")
        pool.register("b")
        pool.register("c")
        assert pool.model_count == 3
        assert len(pool.list_models()) == 3


class TestModelPoolHealthCheck:
    def test_healthy_stays_healthy(self):
        pool = ModelPool()
        pool.register("m1")
        pool.health_check("m1", success=True)
        assert pool.get("m1").status == "healthy"

    def test_healthy_to_degraded_after_two_failures(self):
        pool = ModelPool()
        pool.register("m1")
        pool.health_check("m1", success=False)
        assert pool.get("m1").status == "healthy"  # 1 failure, still healthy
        pool.health_check("m1", success=False)
        assert pool.get("m1").status == "degraded"

    def test_degraded_to_unhealthy_after_four_failures(self):
        pool = ModelPool()
        pool.register("m1")
        for _ in range(4):
            pool.health_check("m1", success=False)
        assert pool.get("m1").status == "unhealthy"

    def test_unhealthy_to_available_after_six_failures(self):
        pool = ModelPool()
        pool.register("m1")
        for _ in range(6):
            pool.health_check("m1", success=False)
        assert pool.get("m1").status == "unavailable"

    def test_recovery_degraded_to_healthy(self):
        pool = ModelPool()
        pool.register("m1")
        for _ in range(2):
            pool.health_check("m1", success=False)
        assert pool.get("m1").status == "degraded"
        # Succeed enough to bring reliability >= 0.9
        for _ in range(10):
            pool.health_check("m1", success=True)
        assert pool.get("m1").status == "healthy"

    def test_recovery_unhealthy_to_degraded(self):
        pool = ModelPool()
        pool.register("m1")
        for _ in range(4):
            pool.health_check("m1", success=False)
        assert pool.get("m1").status == "unhealthy"
        pool.health_check("m1", success=True)
        assert pool.get("m1").status == "degraded"

    def test_health_check_unknown_model_raises(self):
        pool = ModelPool()
        with pytest.raises(ValueError):
            pool.health_check("unknown")

    def test_healthy_models_filter(self):
        pool = ModelPool()
        pool.register("good")
        pool.register("bad")
        for _ in range(6):
            pool.health_check("bad", success=False)
        assert len(pool.healthy_models()) == 1
        assert pool.healthy_count == 1

    def test_degraded_models_filter(self):
        pool = ModelPool()
        pool.register("d1")
        pool.register("d2")
        for _ in range(2):
            pool.health_check("d1", success=False)
        assert len(pool.degraded_models()) == 1

    def test_unhealthy_models_filter(self):
        pool = ModelPool()
        pool.register("u1")
        for _ in range(4):
            pool.health_check("u1", success=False)
        assert len(pool.unhealthy_models()) == 1

    def test_health_check_all(self):
        pool = ModelPool()
        pool.register("a", endpoint="http://a")
        pool.register("b", endpoint="http://b")
        results = pool.health_check_all()
        assert "a" in results
        assert "b" in results

    def test_health_check_all_no_endpoint(self):
        pool = ModelPool()
        pool.register("no_ep")
        results = pool.health_check_all()
        # No endpoint → health check fails once, failure_count=1, still healthy (needs 2 to degrade)
        assert pool.get("no_ep").failure_count == 1


class TestModelPoolRuntimeHelpers:
    def test_record_call_success(self):
        pool = ModelPool()
        pool.register("m1")
        pool.record_call_success("m1")
        assert pool.get("m1").success_count == 1

    def test_record_call_failure(self):
        pool = ModelPool()
        pool.register("m1")
        pool.record_call_failure("m1", "timeout")
        assert pool.get("m1").failure_count == 1
        assert pool.get("m1").last_error == "timeout"

    def test_record_on_unknown_model_noop(self):
        pool = ModelPool()
        pool.record_call_success("unknown")
        pool.record_call_failure("unknown", "err")
        # Should not raise

    def test_reliability_score(self):
        pool = ModelPool()
        pool.register("m1")
        for _ in range(8):
            pool.record_call_success("m1")
        for _ in range(2):
            pool.record_call_failure("m1", "err")
        assert pool.get("m1").reliability_score == pytest.approx(0.8)

    def test_reliability_score_zero_calls(self):
        pool = ModelPool()
        pool.register("m1")
        assert pool.get("m1").reliability_score == 1.0

    def test_select_best_model(self):
        pool = ModelPool()
        pool.register("a")
        pool.register("b")
        for _ in range(9):
            pool.record_call_success("a")
        for _ in range(5):
            pool.record_call_success("b")
        assert pool.select_best_model() == "a"

    def test_select_best_model_raises_when_none_healthy(self):
        pool = ModelPool()
        pool.register("bad")
        for _ in range(6):
            pool.health_check("bad", success=False)
        with pytest.raises(ValueError):
            pool.select_best_model()

    def test_needs_health_check(self):
        pool = ModelPool()
        entry = pool.register("m1", health_check_interval=1.0)
        assert entry.needs_health_check is True
        pool.health_check("m1", success=True)
        # After check, should not need until interval passes
        assert entry.needs_health_check is False
