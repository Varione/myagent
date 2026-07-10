"""
Model Pool — manages model registration, status, and health checks.

A model pool is the source of truth for which models are available at runtime,
their current status (healthy / degraded / unhealthy), and whether they pass
a quick health check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional


@dataclass
class ModelEntry:
    """Single model entry in the pool."""

    model_id: str
    name: str = ""
    provider: str = "local"
    endpoint: str = ""
    api_key: str = ""
    context_length: int = 128_000
    cost_level: float = 0.5
    latency_level: float = 0.5
    status: str = "healthy"  # healthy | degraded | unhealthy | unavailable
    last_health_check_at: Optional[float] = None
    health_check_interval: float = 300.0  # seconds
    failure_count: int = 0
    success_count: int = 0
    last_error: str = ""
    metadata: dict = field(default_factory=dict)

    def is_healthy(self) -> bool:
        return self.status in ("healthy", "degraded")

    @property
    def reliability_score(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    @property
    def needs_health_check(self) -> bool:
        if self.last_health_check_at is None:
            return True
        return (time.time() - self.last_health_check_at) > self.health_check_interval


class ModelPool:
    """
    Registry of available models with status tracking and health checks.

    Usage:
        pool = ModelPool()
        pool.register("local-27b", name="Qwen 27B", endpoint="http://...", provider="local")
        pool.register("remote-gpt4", name="GPT-4", endpoint="https://...", api_key="sk-...")

        healthy = pool.healthy_models()          # list of ModelEntry
        entry = pool.get("local-27b")            # ModelEntry | None
        pool.health_check("local-27b")           # update status based on a ping
    """

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(
        self,
        model_id: str,
        *,
        name: str = "",
        provider: str = "local",
        endpoint: str = "",
        api_key: str = "",
        context_length: int = 128_000,
        cost_level: float = 0.5,
        latency_level: float = 0.5,
        health_check_interval: float = 300.0,
        metadata: Optional[dict] = None,
    ) -> ModelEntry:
        entry = ModelEntry(
            model_id=model_id,
            name=name or model_id,
            provider=provider,
            endpoint=endpoint,
            api_key=api_key,
            context_length=context_length,
            cost_level=cost_level,
            latency_level=latency_level,
            health_check_interval=health_check_interval,
            metadata=metadata or {},
        )
        self._models[model_id] = entry
        return entry

    def unregister(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.pop(model_id, None)

    # ── Lookup ──────────────────────────────────────────────────────

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def list_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def healthy_models(self) -> list[ModelEntry]:
        return [m for m in self._models.values() if m.is_healthy()]

    def degraded_models(self) -> list[ModelEntry]:
        return [m for m in self._models.values() if m.status == "degraded"]

    def unhealthy_models(self) -> list[ModelEntry]:
        return [m for m in self._models.values() if not m.is_healthy()]

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def healthy_count(self) -> int:
        return len(self.healthy_models())

    # ── Health Check ────────────────────────────────────────────────

    def health_check(self, model_id: str, success: bool = True, error_message: str = "") -> ModelEntry:
        """
        Update the health status of a model based on a ping result.

        Args:
            model_id: The model to check.
            success: Whether the ping succeeded.
            error_message: Error string if the ping failed.

        Returns:
            The updated ModelEntry.
        """
        entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model '{model_id}' not found in pool")

        entry.last_health_check_at = time.time()

        if success:
            entry.success_count += 1
            if entry.failure_count > 0:
                entry.failure_count = max(0, entry.failure_count - 1)
            if entry.status == "unhealthy":
                entry.status = "degraded"
            elif entry.status == "degraded" and entry.reliability_score >= 0.9:
                entry.status = "healthy"
        else:
            entry.failure_count += 1
            entry.last_error = error_message or entry.last_error
            if entry.status == "healthy" and entry.failure_count >= 2:
                entry.status = "degraded"
            elif entry.status == "degraded" and entry.failure_count >= 4:
                entry.status = "unhealthy"
            elif entry.failure_count >= 6:
                entry.status = "unavailable"

        return entry

    def health_check_all(self) -> dict[str, str]:
        """
        Run health checks on all models that need one.

        Returns:
            Mapping of model_id → status after check.
        """
        results: dict[str, str] = {}
        for entry in self.list_models():
            if not entry.needs_health_check:
                results[entry.model_id] = entry.status
                continue
            # Default: assume healthy unless endpoint is empty
            success = bool(entry.endpoint)
            self.health_check(entry.model_id, success=success)
            results[entry.model_id] = entry.status
        return results

    # ── Runtime helpers ─────────────────────────────────────────────

    def select_best_model(self, model_ids: Optional[list[str]] = None) -> str:
        """
        Return the model_id with the highest reliability score among the
        given candidates (or all healthy models if none specified).
        """
        candidates = [
            self._models[mid] for mid in (model_ids or self._models)
            if mid in self._models and self._models[mid].is_healthy()
        ]
        if not candidates:
            raise ValueError("No healthy models available")
        return max(candidates, key=lambda m: m.reliability_score).model_id

    def record_call_success(self, model_id: str) -> None:
        entry = self._models.get(model_id)
        if entry:
            entry.success_count += 1

    def record_call_failure(self, model_id: str, error_message: str = "") -> None:
        entry = self._models.get(model_id)
        if entry:
            entry.failure_count += 1
            entry.last_error = error_message or entry.last_error
