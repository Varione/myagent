"""
DR-MMA Phase 2 — CapabilityCalibrator unit tests.

覆盖范围：
  - 记录管理 (record, record_success, record_failure)
  - 校准指标计算 (reliability, avg_latency, ema_reliability)
  - Profile 校准 (calibrate_profile, calibrate_all)
  - 窗口大小限制、清除与摘要功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.engine.capability_calibrator import CapabilityCalibrator, CalibrationRecord
from dr_mma.engine.capabilities import CapabilityProfile


class TestRecordManagement:
    def test_record_success(self):
        cal = CapabilityCalibrator()
        rec = cal.record("m1", success=True, latency_ms=100)
        assert rec.success is True
        assert cal.record_count("m1") == 1

    def test_record_failure(self):
        cal = CapabilityCalibrator()
        cal.record_failure("m1", error_message="timeout")
        assert cal.record_count("m1") == 1
        assert cal.get_records("m1")[0].error_message == "timeout"

    def test_record_success_convenience(self):
        cal = CapabilityCalibrator()
        cal.record_success("m1", latency_ms=50, confidence=0.9)
        rec = cal.get_records("m1")[0]
        assert rec.success is True
        assert rec.latency_ms == 50
        assert rec.confidence == 0.9

    def test_record_count_unknown_model(self):
        cal = CapabilityCalibrator()
        assert cal.record_count("unknown") == 0

    def test_window_size_trimming(self):
        cal = CapabilityCalibrator(window_size=3)
        for i in range(5):
            cal.record("m1", success=True)
        assert cal.record_count("m1") == 3

    def test_get_records_empty(self):
        cal = CapabilityCalibrator()
        assert cal.get_records("m1") == []


class TestCalibrationMetrics:
    def test_compute_reliability_all_success(self):
        cal = CapabilityCalibrator()
        for _ in range(10):
            cal.record_success("m1")
        assert cal.compute_reliability("m1") == pytest.approx(1.0)

    def test_compute_reliability_mixed(self):
        cal = CapabilityCalibrator()
        for _ in range(8):
            cal.record_success("m1")
        for _ in range(2):
            cal.record_failure("m1")
        assert cal.compute_reliability("m1") == pytest.approx(0.8)

    def test_compute_reliability_none_for_unknown(self):
        cal = CapabilityCalibrator()
        assert cal.compute_reliability("unknown") is None

    def test_compute_avg_latency(self):
        cal = CapabilityCalibrator()
        cal.record("m1", success=True, latency_ms=100)
        cal.record("m1", success=True, latency_ms=200)
        assert cal.compute_avg_latency("m1") == pytest.approx(150.0)

    def test_compute_avg_confidence(self):
        cal = CapabilityCalibrator()
        cal.record("m1", success=True, confidence=0.9)
        cal.record("m1", success=True, confidence=0.7)
        assert cal.compute_avg_confidence("m1") == pytest.approx(0.8)

    def test_compute_ema_reliability(self):
        cal = CapabilityCalibrator(alpha=0.2)
        for _ in range(5):
            cal.record_success("m1")
        ema = cal.compute_ema_reliability("m1")
        assert ema is not None
        assert ema > 0.9  # all successes → high EMA

    def test_compute_ema_reliability_recent_failures_drop_score(self):
        cal = CapabilityCalibrator(alpha=0.3)
        for _ in range(10):
            cal.record_success("m1")
        for _ in range(5):
            cal.record_failure("m1")
        ema = cal.compute_ema_reliability("m1")
        assert ema is not None
        assert ema < 0.8  # recent failures should drag score down


class TestProfileCalibration:
    def test_calibrate_profile_below_min_samples(self):
        cal = CapabilityCalibrator(min_samples=10)
        for _ in range(5):
            cal.record_success("m1")
        profile = CapabilityProfile(model_name="m1", historical_score=0.7, reliability=0.9)
        result = cal.calibrate_profile(profile)
        assert result.update_source == "default"
        # Should not change scores below threshold
        assert result.historical_score == 0.7

    def test_calibrate_profile_updates_scores(self):
        cal = CapabilityCalibrator(min_samples=3, alpha=0.2)
        for _ in range(8):
            cal.record_success("m1", latency_ms=200, confidence=0.9)
        profile = CapabilityProfile(
            model_name="m1",
            historical_score=0.7,
            reliability=0.9,
            latency_score=0.8,
            confidence=0.5,
        )
        result = cal.calibrate_profile(profile)
        assert result.update_source == "history"
        assert result.sample_count == 8
        assert result.reliability > 0.9  # EMA should push toward 1.0

    def test_calibrate_all(self):
        cal = CapabilityCalibrator(min_samples=2)
        for _ in range(5):
            cal.record_success("m1")
            cal.record_success("m2")
        profiles = [
            CapabilityProfile(model_name="m1", historical_score=0.7),
            CapabilityProfile(model_name="m2", historical_score=0.7),
        ]
        results = cal.calibrate_all(profiles)
        assert len(results) == 2
        assert all(r.update_source == "history" for r in results)


class TestClearAndSummary:
    def test_clear_specific_model(self):
        cal = CapabilityCalibrator()
        cal.record_success("m1")
        cal.record_success("m2")
        cal.clear("m1")
        assert cal.record_count("m1") == 0
        assert cal.record_count("m2") == 1

    def test_clear_all(self):
        cal = CapabilityCalibrator()
        cal.record_success("m1")
        cal.record_success("m2")
        cal.clear()
        assert cal.record_count("m1") == 0
        assert cal.record_count("m2") == 0

    def test_summary_with_data(self):
        cal = CapabilityCalibrator()
        for _ in range(8):
            cal.record_success("m1", latency_ms=100, confidence=0.9)
        for _ in range(2):
            cal.record("m1", success=False, latency_ms=500, confidence=0.3)
        s = cal.summary("m1")
        assert s["record_count"] == 10
        assert s["reliability"] == pytest.approx(0.8)
        assert s["failure_count"] == 2

    def test_summary_empty_model(self):
        cal = CapabilityCalibrator()
        s = cal.summary("unknown")
        assert s["record_count"] == 0
        assert s["reliability"] is None


class TestEMAUpdate:
    def test_ema_update_basic(self):
        result = CapabilityCalibrator._ema_update(0.7, 1.0, alpha=0.2)
        # 0.2 * 1.0 + 0.8 * 0.7 = 0.2 + 0.56 = 0.76
        assert result == pytest.approx(0.76)

    def test_ema_update_alpha_zero(self):
        result = CapabilityCalibrator._ema_update(0.7, 1.0, alpha=0.0)
        assert result == pytest.approx(0.7)

    def test_ema_update_alpha_one(self):
        result = CapabilityCalibrator._ema_update(0.7, 1.0, alpha=1.0)
        assert result == pytest.approx(1.0)
