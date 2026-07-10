"""
CapabilityCalibrator — runtime capability calibration based on call history.

Phase 2: dynamically adjusts CapabilityProfile scores as models accumulate
runtime call data (successes, failures, latency, etc.). The calibrator provides
a feedback loop so that historical performance gradually replaces static
benchmark scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .capabilities import CapabilityProfile


@dataclass
class CalibrationRecord:
    """Single calibration observation."""

    model_id: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    latency_ms: float = 0.0
    task_type: str = "general"  # planning | execution | review | verification | summary
    confidence: float = 1.0     # model self-reported confidence
    error_message: str = ""

    @property
    def outcome_score(self) -> float:
        """Return 1.0 for success, 0.0 for failure."""
        return 1.0 if self.success else 0.0


class CapabilityCalibrator:
    """
    Calibrates model capability scores based on runtime call history.

    The calibrator maintains a rolling window of CalibrationRecords per model
    and periodically updates the CapabilityProfile in the registry with
    calibrated scores derived from actual performance data.

    Key features:
    - Exponential moving average (EMA) for smooth score transitions
    - Separate calibration for reliability, latency, and cost dimensions
    - Minimum sample threshold before overriding benchmark scores
    - Decay factor to weight recent observations more heavily
    """

    def __init__(
        self,
        alpha: float = 0.15,
        min_samples: int = 5,
        window_size: int = 100,
    ):
        """
        Args:
            alpha: EMA smoothing factor (0-1). Higher = faster adaptation.
            min_samples: Minimum observations before calibration kicks in.
            window_size: Maximum number of records kept per model.
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.window_size = window_size
        self._records: dict[str, list[CalibrationRecord]] = {}

    # ── Record management ───────────────────────────────────────────

    def record(
        self,
        model_id: str,
        success: bool = True,
        latency_ms: float = 0.0,
        task_type: str = "general",
        confidence: float = 1.0,
        error_message: str = "",
    ) -> CalibrationRecord:
        """Record a single model call observation."""
        rec = CalibrationRecord(
            model_id=model_id,
            success=success,
            latency_ms=latency_ms,
            task_type=task_type,
            confidence=confidence,
            error_message=error_message,
        )
        self._records.setdefault(model_id, []).append(rec)
        # Trim to window size
        records = self._records[model_id]
        if len(records) > self.window_size:
            self._records[model_id] = records[-self.window_size:]
        return rec

    def record_success(
        self,
        model_id: str,
        latency_ms: float = 0.0,
        task_type: str = "general",
        confidence: float = 1.0,
    ) -> CalibrationRecord:
        """Convenience: record a successful call."""
        return self.record(model_id, success=True, latency_ms=latency_ms, task_type=task_type, confidence=confidence)

    def record_failure(
        self,
        model_id: str,
        latency_ms: float = 0.0,
        task_type: str = "general",
        error_message: str = "",
    ) -> CalibrationRecord:
        """Convenience: record a failed call."""
        return self.record(
            model_id, success=False, latency_ms=latency_ms, task_type=task_type, error_message=error_message
        )

    def get_records(self, model_id: str) -> list[CalibrationRecord]:
        """Return all records for a model."""
        return list(self._records.get(model_id, []))

    def record_count(self, model_id: str) -> int:
        """Return the number of records for a model."""
        return len(self._records.get(model_id, []))

    # ── Calibration metrics ─────────────────────────────────────────

    def compute_reliability(self, model_id: str) -> Optional[float]:
        """Compute reliability score (success rate) from recent records."""
        records = self._records.get(model_id, [])
        if not records:
            return None
        successes = sum(1 for r in records if r.success)
        return successes / len(records)

    def compute_avg_latency(self, model_id: str) -> Optional[float]:
        """Compute average latency (ms) from recent records."""
        records = self._records.get(model_id, [])
        if not records:
            return None
        return sum(r.latency_ms for r in records) / len(records)

    def compute_avg_confidence(self, model_id: str) -> Optional[float]:
        """Compute average self-reported confidence from recent records."""
        records = self._records.get(model_id, [])
        if not records:
            return None
        return sum(r.confidence for r in records) / len(records)

    def compute_ema_reliability(self, model_id: str) -> Optional[float]:
        """
        Compute EMA reliability score, weighting recent observations more.

        Uses exponential decay so the most recent N calls dominate.
        """
        records = self._records.get(model_id, [])
        if not records:
            return None

        ema = 0.0
        weight_sum = 0.0
        for i, rec in enumerate(records):
            # Decay factor: older records get exponentially less weight
            decay = (1 - self.alpha) ** (len(records) - 1 - i)
            weight = self.alpha * decay + (1 if i == len(records) - 1 else 0)
            ema += rec.outcome_score * weight
            weight_sum += weight

        return ema / weight_sum if weight_sum > 0 else None

    # ── Profile calibration ─────────────────────────────────────────

    def calibrate_profile(self, profile: CapabilityProfile) -> CapabilityProfile:
        """
        Update a CapabilityProfile with calibrated scores from runtime data.

        Uses EMA-based reliability and latency to adjust the profile's
        historical_score, reliability, and latency_score fields.

        Only applies calibration when min_samples threshold is met.
        """
        records = self._records.get(profile.model_name, [])
        n = len(records)

        if n < self.min_samples:
            profile.update_source = "default"
            return profile

        # Calibrate reliability
        ema_rel = self.compute_ema_reliability(profile.model_name)
        if ema_rel is not None:
            profile.reliability = self._ema_update(profile.reliability, ema_rel)

        # Calibrate historical score from success rate
        rel = self.compute_reliability(profile.model_name)
        if rel is not None:
            profile.historical_score = self._ema_update(profile.historical_score, rel)

        # Calibrate latency score (normalize avg latency to 0-1 scale)
        avg_lat = self.compute_avg_latency(profile.model_name)
        if avg_lat is not None:
            # Map latency to score: <500ms → 1.0, >5000ms → 0.1
            lat_score = max(0.1, min(1.0, 1.0 - (avg_lat - 500) / 4500))
            profile.latency_score = self._ema_update(profile.latency_score, lat_score)

        # Update confidence from model self-reports
        avg_conf = self.compute_avg_confidence(profile.model_name)
        if avg_conf is not None:
            profile.confidence = self._ema_update(profile.confidence, avg_conf)

        # Track sample count and failure count
        profile.sample_count = n
        profile.failure_count = sum(1 for r in records if not r.success)
        profile.last_evaluated_at = time.time()
        profile.update_source = "history"

        return profile

    def calibrate_all(self, profiles: list[CapabilityProfile]) -> list[CapabilityProfile]:
        """Calibrate multiple profiles at once."""
        return [self.calibrate_profile(p) for p in profiles]

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _ema_update(current: float, new_value: float, alpha: float = 0.15) -> float:
        """Standard EMA update: new = alpha * value + (1-alpha) * current."""
        return alpha * new_value + (1 - alpha) * current

    # ── Reset / diagnostics ─────────────────────────────────────────

    def clear(self, model_id: Optional[str] = None):
        """Clear records for a specific model or all models."""
        if model_id is None:
            self._records.clear()
        else:
            self._records.pop(model_id, None)

    def summary(self, model_id: str) -> dict:
        """Return a diagnostic summary for a model."""
        records = self._records.get(model_id, [])
        if not records:
            return {
                "model_id": model_id,
                "record_count": 0,
                "reliability": None,
                "avg_latency_ms": None,
                "avg_confidence": None,
            }
        return {
            "model_id": model_id,
            "record_count": len(records),
            "reliability": self.compute_reliability(model_id),
            "ema_reliability": self.compute_ema_reliability(model_id),
            "avg_latency_ms": round(self.compute_avg_latency(model_id) or 0, 1),
            "avg_confidence": round(self.compute_avg_confidence(model_id) or 0, 3),
            "failure_count": sum(1 for r in records if not r.success),
        }
