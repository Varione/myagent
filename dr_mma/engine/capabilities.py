"""Capability registry and dynamic role assignment."""

from dataclasses import dataclass, field
from typing import Optional


ROLE_CAPABILITIES = {
    "Supervisor": ["reasoning", "synthesis", "decision"],
    "Planner": ["planning", "reasoning"],
    "Researcher": ["research", "tool_use"],
    "Domain Expert": ["domain_knowledge", "reasoning"],
    "Worker": ["coding", "tool_use"],
    "Critic": ["critic", "reasoning"],
    "Verifier": ["verification", "reasoning"],
    "Synthesizer": ["synthesis", "writing"],
}


@dataclass
class CapabilityProfile:
    model_name: str
    benchmark_score: float = 0.7
    historical_score: float = 0.7
    human_feedback_score: float = 0.7
    failure_penalty: float = 0.0
    reliability: float = 0.9
    latency_score: float = 0.8
    cost_score: float = 0.8
    tool_score: float = 0.8
    capabilities: dict[str, float] = field(default_factory=dict)

    @property
    def calibrated_score(self) -> float:
        score = (
            0.4 * self.benchmark_score
            + 0.4 * self.historical_score
            + 0.2 * self.human_feedback_score
            - self.failure_penalty
        )
        return max(0.0, min(1.0, score))


class CapabilityRegistry:
    """Stores model capability profiles and selects the best fit per role."""

    def __init__(self):
        self._profiles: dict[str, CapabilityProfile] = {}

    def ensure_profile(self, model_name: str) -> CapabilityProfile:
        if model_name not in self._profiles:
            self._profiles[model_name] = self._build_default_profile(model_name)
        return self._profiles[model_name]

    def register(self, profile: CapabilityProfile):
        self._profiles[profile.model_name] = profile

    def list_profiles(self) -> list[CapabilityProfile]:
        return list(self._profiles.values())

    def select_model_for_role(self, role: str, available_models: list[str], exclude: Optional[set[str]] = None) -> str:
        if not available_models:
            raise ValueError("No available models for dynamic role assignment")
        excluded = exclude or set()
        ranked: list[tuple[float, str]] = []
        for model_name in available_models:
            if model_name in excluded:
                continue
            profile = self.ensure_profile(model_name)
            ranked.append((self._score_role_fit(profile, role), model_name))
        if not ranked:
            return available_models[0]
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _score_role_fit(self, profile: CapabilityProfile, role: str) -> float:
        capabilities = ROLE_CAPABILITIES.get(role, [])
        if capabilities:
            capability_fit = sum(profile.capabilities.get(cap, profile.calibrated_score) for cap in capabilities) / len(capabilities)
        else:
            capability_fit = profile.calibrated_score
        return (
            0.35 * capability_fit
            + 0.20 * profile.reliability
            + 0.20 * profile.tool_score
            + 0.10 * profile.calibrated_score
            - 0.10 * (1.0 - profile.cost_score)
            - 0.05 * (1.0 - profile.latency_score)
        )

    def _build_default_profile(self, model_name: str) -> CapabilityProfile:
        lower = model_name.lower()
        base = CapabilityProfile(model_name=model_name)
        base.capabilities = {
            "reasoning": 0.8,
            "planning": 0.8,
            "tool_use": 0.8,
            "coding": 0.75,
            "critic": 0.8,
            "verification": 0.8,
            "synthesis": 0.8,
            "writing": 0.75,
            "research": 0.75,
            "domain_knowledge": 0.75,
            "decision": 0.8,
        }
        if "mock" in lower:
            base.benchmark_score = 0.55
            base.historical_score = 0.6
            base.human_feedback_score = 0.6
            base.reliability = 0.95
            base.latency_score = 0.95
            base.cost_score = 1.0
        if "remote" in lower:
            base.tool_score = 0.75
            base.cost_score = 0.6
        if "local" in lower:
            base.latency_score = 0.9
            base.cost_score = 0.9
        return base


class DynamicRoleAssigner:
    """Assigns models to runtime roles based on capability profiles."""

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def assign(self, roles: list[str], available_models: list[str]) -> dict[str, str]:
        assignments: dict[str, str] = {}
        used: set[str] = set()
        priority = ["Supervisor", "Critic", "Verifier", "Planner", "Worker", "Researcher", "Domain Expert", "Synthesizer"]
        ordered = sorted(roles, key=lambda role: priority.index(role) if role in priority else len(priority))
        for role in ordered:
            model = self.registry.select_model_for_role(role, available_models, exclude=used if len(available_models) > len(used) else set())
            assignments[role] = model
            used.add(model)
        return assignments
