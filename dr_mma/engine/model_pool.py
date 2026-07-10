"""
Model Pool — manages model registration, status, and health checks.

A model pool is the source of truth for which models are available at runtime,
their current status (healthy / degraded / unhealthy), and whether they pass
a quick health check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import socket
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

    @staticmethod
    def _probe_endpoint(entry: ModelEntry) -> tuple[bool, str]:
        """
        Probe a model endpoint based on its type.

        Returns:
            (success, error_message)
        """
        provider = entry.provider.lower()

        # Mock / in-memory models are always healthy
        if provider in ("mock", "in_memory", "memory"):
            return True, ""

        endpoint = entry.endpoint.strip()
        if not endpoint:
            return False, "endpoint is empty"

        # Local process: check TCP port connectivity
        if provider in ("local", "local_process", "ollama", "lm_studio"):
            try:
                host_port = endpoint.replace("http://", "").replace("https://", "")
                if "/" in host_port:
                    host_port = host_port.split("/")[0]
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((host, port))
                return True, ""
            except Exception as e:
                return False, f"local_connection_failed: {e}"

        # Remote HTTP: send lightweight HEAD request
        if provider in ("remote", "remote_http", "openai", "anthropic", "azure"):
            try:
                parsed = endpoint.split("/v")[0].split("/chat")[0].split("/completions")[0]
                scheme = "https" if parsed.startswith("https") else "http"
                host_port = parsed.replace("https://", "").replace("http://", "")
                if "/" in host_port:
                    host, path = host_port.split("/", 1)
                    path = "/" + path
                else:
                    host = host_port
                    path = "/v1/models"

                if scheme == "https":
                    conn = http.client.HTTPSConnection(host, timeout=5)
                else:
                    conn = http.client.HTTPConnection(host, timeout=5)
                try:
                    headers = {}
                    if entry.api_key:
                        headers["Authorization"] = f"Bearer {entry.api_key[:8]}..."
                    conn.request("HEAD", path, headers=headers)
                    resp = conn.getresponse()
                    # 2xx and 4xx are acceptable (401 means endpoint reachable but key invalid)
                    if resp.status in (200, 204, 400, 401, 403):
                        return True, ""
                    return False, f"http_{resp.status}"
                finally:
                    conn.close()
            except Exception as e:
                return False, f"http_probe_failed: {e}"

        # Unknown provider: fall back to endpoint existence check
        return bool(endpoint), ""

    def health_check_all(self) -> dict[str, str]:
        """
        Run real health checks on all models that need one.

        Probes based on provider type:
        - mock/in_memory: always healthy
        - local/local_process: TCP port check
        - remote/remote_http: HTTP HEAD request

        Returns:
            Mapping of model_id → status after check.
        """
        results: dict[str, str] = {}
        for entry in self.list_models():
            if not entry.needs_health_check:
                results[entry.model_id] = entry.status
                continue
            success, error_msg = self._probe_endpoint(entry)
            self.health_check(entry.model_id, success=success, error_message=error_msg)
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
