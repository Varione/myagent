"""
Dynamic Role Management — Role Merge, Split, and Failover.

Phase 2 of DR-MMA: roles can be dynamically merged (multiple roles assigned
to a single model) or split (one role's work distributed across multiple
models), and failover is handled when models become unhealthy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..engine.capabilities import CapabilityProfile, CapabilityRegistry, ROLE_CAPABILITIES
from ..engine.model_pool import ModelPool


# ── Merge / Split Configuration ─────────────────────────────────────

@dataclass
class MergeConfig:
    """Controls when and how roles can be merged onto a single model."""
    max_roles_per_model: int = 3
    min_capability_overlap: float = 0.6
    min_reliability_threshold: float = 0.85
    cost_savings_threshold: float = 0.15  # must save at least this much

@dataclass
class SplitConfig:
    """Controls when and how a role can be split across models."""
    min_failure_count_to_split: int = 3
    max_models_per_role: int = 3
    min_confidence_drop: float = 0.2


# ── Role Binding State ─────────────────────────────────────────────

@dataclass
class RoleBinding:
    """Tracks which model(s) are bound to a role at runtime."""
    role: str
    primary_model: str
    backup_models: list[str] = field(default_factory=list)
    split_assignments: dict[str, str] = field(default_factory=dict)  # sub-task → model
    is_merged: bool = False
    merged_into: Optional[str] = None
    confidence: float = 0.8
    last_updated_at: Optional[float] = None

    def __post_init__(self):
        if self.last_updated_at is None:
            self.last_updated_at = time.time()


# ── Role Merger / Splitter ─────────────────────────────────────────

class RoleMergerSplitter:
    """
    Decides whether to merge or split roles based on model pool state,
    capability profiles, and runtime configuration.
    """

    def __init__(
        self,
        pool: ModelPool,
        registry: CapabilityRegistry,
        merge_config: Optional[MergeConfig] = None,
        split_config: Optional[SplitConfig] = None,
    ):
        self.pool = pool
        self.registry = registry
        self.merge_config = merge_config or MergeConfig()
        self.split_config = split_config or SplitConfig()
        self._bindings: dict[str, RoleBinding] = {}

    # ── Bindings ────────────────────────────────────────────────────

    def get_binding(self, role: str) -> Optional[RoleBinding]:
        return self._bindings.get(role)

    def set_binding(self, binding: RoleBinding):
        self._bindings[binding.role] = binding

    def list_bindings(self) -> list[RoleBinding]:
        return list(self._bindings.values())

    # ── Merge Decision ─────────────────────────────────────────────

    def should_merge(self, role_a: str, role_b: str) -> bool:
        """
        Return True if two roles can be safely merged onto the same model.

        Criteria:
        1. Capability overlap is above threshold (they share similar needs).
        2. A single healthy model exists that can handle both.
        3. Merging saves cost above the threshold.
        4. Model hasn't exceeded max_roles_per_model.
        """
        cap_a = set(ROLE_CAPABILITIES.get(role_a, []))
        cap_b = set(ROLE_CAPABILITIES.get(role_b, []))
        overlap = len(cap_a & cap_b) / max(len(cap_a | cap_b), 1)

        if overlap < self.merge_config.min_capability_overlap:
            return False

        # Find a model that can handle both
        best_model = self._find_best_combined_model(role_a, role_b)
        if best_model is None:
            return False

        profile = self.registry.ensure_profile(best_model)
        if profile.reliability < self.merge_config.min_reliability_threshold:
            return False

        # Check model load
        current_load = sum(
            1 for b in self._bindings.values()
            if b.primary_model == best_model and not b.is_merged
        )
        if current_load >= self.merge_config.max_roles_per_model:
            return False

        # Cost savings check
        cost_savings = self._estimate_cost_savings(role_a, role_b, best_model)
        return cost_savings >= self.merge_config.cost_savings_threshold

    def merge_roles(self, role_a: str, role_b: str) -> Optional[RoleBinding]:
        """
        Attempt to merge two roles into a single model binding.
        Returns the new merged RoleBinding or None if merge fails.
        """
        if not self.should_merge(role_a, role_b):
            return None

        best_model = self._find_best_combined_model(role_a, role_b)
        if best_model is None:
            return None

        # Determine which role survives as primary
        primary_role = role_a
        secondary_role = role_b

        binding = RoleBinding(
            role=primary_role,
            primary_model=best_model,
            is_merged=True,
            merged_into=secondary_role,
            confidence=self._combined_confidence(role_a, role_b, best_model),
        )

        # Remove old bindings
        self._bindings.pop(role_a, None)
        self._bindings.pop(role_b, None)

        self._bindings[primary_role] = binding
        return binding

    # ── Split Decision ─────────────────────────────────────────────

    def should_split(self, role: str) -> bool:
        """
        Return True if a role's primary model is underperforming and
        splitting the work across multiple models would help.
        """
        binding = self._bindings.get(role)
        if binding is None:
            return False

        profile = self.registry.ensure_profile(binding.primary_model)
        if profile.failure_count < self.split_config.min_failure_count_to_split:
            return False

        if profile.confidence > (1.0 - self.split_config.min_confidence_drop):
            return False

        # Check we have enough healthy models to split into
        healthy = self.pool.healthy_models()
        return len(healthy) >= 2

    def split_role(self, role: str) -> Optional[RoleBinding]:
        """
        Split a role across multiple models based on sub-task distribution.
        Returns the updated RoleBinding or None if split fails.
        """
        binding = self._bindings.get(role)
        if binding is None or not self.should_split(role):
            return None

        healthy_models = self.pool.healthy_models()
        if len(healthy_models) < 2:
            return None

        # Select top N models for this role's capabilities
        selected = []
        for entry in healthy_models[:self.split_config.max_models_per_role]:
            selected.append(entry.model_id)

        binding.is_merged = False
        binding.merged_into = None
        binding.backup_models = selected[1:]
        binding.split_assignments = {
            f"subtask_{i}": model for i, model in enumerate(selected)
        }
        binding.confidence = min(
            self.registry.ensure_profile(m).calibrated_score
            for m in selected
        )
        binding.last_updated_at = time.time()

        return binding

    # ── Failover ───────────────────────────────────────────────────

    def handle_failover(self, model_id: str) -> list[str]:
        """
        When a model fails, reassign all roles bound to it.
        Returns a list of affected role names.
        """
        entry = self.pool.get(model_id)
        if entry is None:
            return []

        self.pool.health_check(model_id, success=False)

        affected: list[str] = []
        for role, binding in self._bindings.items():
            if binding.primary_model != model_id:
                continue

            # Try backup models first
            new_model = None
            for backup in binding.backup_models:
                if self.pool.get(backup) and self.pool.get(backup).is_healthy():
                    new_model = backup
                    break

            # If no backup, find a healthy replacement
            if new_model is None:
                new_model = self._find_replacement(role, exclude={model_id})

            if new_model is None:
                affected.append(role)
                continue

            binding.primary_model = new_model
            binding.confidence *= 0.9  # slight confidence drop on failover
            binding.last_updated_at = time.time()
            affected.append(role)

        return affected

    # ── Internal Helpers ───────────────────────────────────────────

    def _find_best_combined_model(self, role_a: str, role_b: str) -> Optional[str]:
        """Find the single best model that can handle both roles."""
        healthy = self.pool.healthy_models()
        if not healthy:
            return None

        candidates: list[tuple[float, str]] = []
        for entry in healthy:
            profile = self.registry.ensure_profile(entry.model_id)
            score_a = self._score_role_fit(profile, role_a)
            score_b = self._score_role_fit(profile, role_b)
            combined = (score_a + score_b) / 2.0
            candidates.append((combined, entry.model_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _find_replacement(self, role: str, exclude: set[str]) -> Optional[str]:
        """Find the best replacement model for a given role."""
        try:
            return self.registry.select_model_for_role(
                role,
                [e.model_id for e in self.pool.healthy_models() if e.model_id not in exclude],
                exclude=exclude,
            )
        except (ValueError, IndexError):
            return None

    def _score_role_fit(self, profile: CapabilityProfile, role: str) -> float:
        """Score how well a model fits a role (mirrors registry logic)."""
        capabilities = ROLE_CAPABILITIES.get(role, [])
        if capabilities:
            cap_fit = sum(
                profile.capabilities.get(c, profile.calibrated_score)
                for c in capabilities
            ) / len(capabilities)
        else:
            cap_fit = profile.calibrated_score

        return (
            0.35 * cap_fit
            + 0.20 * profile.reliability
            + 0.20 * profile.tool_score
            + 0.10 * profile.calibrated_score
            - 0.10 * (1.0 - profile.cost_score)
            - 0.05 * (1.0 - profile.latency_score)
        )

    def _combined_confidence(self, role_a: str, role_b: str, model_id: str) -> float:
        profile = self.registry.ensure_profile(model_id)
        score_a = self._score_role_fit(profile, role_a)
        score_b = self._score_role_fit(profile, role_b)
        return min(score_a, score_b) * profile.reliability

    def _estimate_cost_savings(
        self, role_a: str, role_b: str, combined_model: str
    ) -> float:
        """
        Rough estimate of cost savings from merging.
        Returns a value in [0, 1] where higher means more savings.
        """
        profile = self.registry.ensure_profile(combined_model)
        # Merging saves one model's cost — normalize by cost_score
        return 1.0 - profile.cost_score
