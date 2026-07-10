"""
DR-MMA Phase 2 — RoleMergerSplitter unit tests.

覆盖范围：
  - 绑定管理 (get/set/list)
  - should_merge / merge_roles 决策与执行
  - should_split / split_role 决策与执行
  - handle_failover 故障转移
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.engine.model_pool import ModelPool
from dr_mma.engine.capabilities import CapabilityRegistry, CapabilityProfile
from dr_mma.engine.role_manager import (
    RoleMergerSplitter,
    RoleBinding,
    MergeConfig,
    SplitConfig,
)


def _make_pool_with_models(count: int = 4):
    """Create a pool with count healthy models."""
    pool = ModelPool()
    for i in range(count):
        pool.register(f"m{i}", name=f"Model {i}")
    return pool


def _make_registry_with_profiles(pool: ModelPool):
    """Create a registry with good profiles for all models in the pool."""
    reg = CapabilityRegistry()
    for entry in pool.list_models():
        profile = CapabilityProfile(
            model_name=entry.model_id,
            benchmark_score=0.9,
            historical_score=0.9,
            human_feedback_score=0.9,
            reliability=0.95,
            latency_score=0.9,
            cost_score=0.8,
            tool_score=0.9,
            confidence=0.85,
            sample_count=20,
        )
        profile.capabilities = {
            "reasoning": 0.9,
            "planning": 0.9,
            "tool_use": 0.9,
            "coding": 0.9,
            "critic": 0.9,
            "verification": 0.9,
            "synthesis": 0.9,
            "writing": 0.9,
            "research": 0.9,
            "domain_knowledge": 0.9,
            "decision": 0.9,
        }
        reg.register(profile)
    return reg


class TestRoleBindingManagement:
    def test_set_and_get_binding(self):
        pool = _make_pool_with_models(2)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        binding = RoleBinding(role="Planner", primary_model="m0")
        rms.set_binding(binding)
        assert rms.get_binding("Planner").primary_model == "m0"

    def test_get_nonexistent_binding(self):
        pool = _make_pool_with_models(2)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        assert rms.get_binding("Planner") is None

    def test_list_bindings(self):
        pool = _make_pool_with_models(2)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Planner", primary_model="m0"))
        rms.set_binding(RoleBinding(role="Worker", primary_model="m1"))
        assert len(rms.list_bindings()) == 2


class TestMergeDecision:
    def test_should_merge_returns_true_when_overlap_high(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        # Senior Supervisor & Junior Supervisor share reasoning/synthesis/decision → overlap > 0.6
        assert rms.should_merge("Senior Supervisor", "Junior Supervisor") is True

    def test_should_merge_returns_false_when_overlap_low(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        # Researcher and Verifier have very different capabilities
        assert rms.should_merge("Researcher", "Verifier") is False

    def test_should_merge_false_when_no_healthy_models(self):
        pool = ModelPool()
        reg = CapabilityRegistry()
        rms = RoleMergerSplitter(pool, reg)
        assert rms.should_merge("Planner", "Worker") is False

    def test_should_merge_false_when_model_exceeds_max_roles(self):
        pool = _make_pool_with_models(2)
        reg = _make_registry_with_profiles(pool)
        # Set max_roles_per_model=1 so merging any two roles exceeds it
        rms = RoleMergerSplitter(
            pool, reg, merge_config=MergeConfig(max_roles_per_model=1)
        )
        assert rms.should_merge("Planner", "Worker") is False

    def test_should_merge_false_when_reliability_below_threshold(self):
        pool = _make_pool_with_models(3)
        reg = CapabilityRegistry()
        # All models have low reliability
        for i in range(3):
            profile = CapabilityProfile(
                model_name=f"m{i}", reliability=0.5, benchmark_score=0.5,
                historical_score=0.5, human_feedback_score=0.5,
            )
            reg.register(profile)
        rms = RoleMergerSplitter(pool, reg)
        assert rms.should_merge("Planner", "Worker") is False

    def test_merge_roles_succeeds(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        result = rms.merge_roles("Senior Supervisor", "Junior Supervisor")
        assert result is not None
        assert result.is_merged is True
        assert result.merged_into == "Junior Supervisor"

    def test_merge_roles_returns_none_when_should_merge_fails(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        # Researcher and Verifier have low overlap
        result = rms.merge_roles("Researcher", "Verifier")
        assert result is None

    def test_merge_removes_old_bindings(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Senior Supervisor", primary_model="m0"))
        rms.set_binding(RoleBinding(role="Junior Supervisor", primary_model="m1"))
        rms.merge_roles("Senior Supervisor", "Junior Supervisor")
        assert rms.get_binding("Junior Supervisor") is None  # secondary removed


class TestSplitDecision:
    def test_should_split_returns_true_when_failures_high(self):
        pool = _make_pool_with_models(4)
        reg = CapabilityRegistry()
        profile = CapabilityProfile(
            model_name="m0", benchmark_score=0.9, historical_score=0.9,
            human_feedback_score=0.9, reliability=0.95,
            confidence=0.3, sample_count=30, failure_count=5,
        )
        reg.register(profile)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="m0"))
        assert rms.should_split("Worker") is True

    def test_should_split_returns_false_when_no_binding(self):
        pool = _make_pool_with_models(4)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        assert rms.should_split("Worker") is False

    def test_should_split_returns_false_when_confidence_high(self):
        pool = _make_pool_with_models(4)
        reg = CapabilityRegistry()
        profile = CapabilityProfile(
            model_name="m0", benchmark_score=0.9, historical_score=0.9,
            human_feedback_score=0.9, reliability=0.95,
            confidence=0.95, sample_count=30, failure_count=5,
        )
        reg.register(profile)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="m0"))
        assert rms.should_split("Worker") is False

    def test_should_split_returns_false_when_not_enough_healthy_models(self):
        pool = ModelPool()
        pool.register("only_one")
        reg = CapabilityRegistry()
        profile = CapabilityProfile(
            model_name="only_one", benchmark_score=0.9, historical_score=0.9,
            human_feedback_score=0.9, reliability=0.95,
            confidence=0.3, sample_count=30, failure_count=5,
        )
        reg.register(profile)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="only_one"))
        assert rms.should_split("Worker") is False

    def test_split_role_succeeds(self):
        pool = _make_pool_with_models(4)
        reg = CapabilityRegistry()
        profile = CapabilityProfile(
            model_name="m0", benchmark_score=0.9, historical_score=0.9,
            human_feedback_score=0.9, reliability=0.95,
            confidence=0.3, sample_count=30, failure_count=5,
        )
        reg.register(profile)
        for i in range(1, 4):
            p = CapabilityProfile(
                model_name=f"m{i}", benchmark_score=0.9, historical_score=0.9,
                human_feedback_score=0.9, reliability=0.95,
                confidence=0.85, sample_count=20,
            )
            reg.register(p)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="m0"))
        result = rms.split_role("Worker")
        assert result is not None
        assert len(result.backup_models) >= 1
        assert len(result.split_assignments) >= 2

    def test_split_role_returns_none_when_should_split_fails(self):
        pool = _make_pool_with_models(4)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="m0"))
        # No failures → should not split
        result = rms.split_role("Worker")
        assert result is None


class TestFailover:
    def test_failover_reassigns_from_backup(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(
            role="Worker", primary_model="m0", backup_models=["m1"]
        ))
        # Make m0 unhealthy
        for _ in range(6):
            pool.health_check("m0", success=False)
        affected = rms.handle_failover("m0")
        assert "Worker" in affected
        assert rms.get_binding("Worker").primary_model == "m1"

    def test_failover_finds_replacement_when_no_backup(self):
        pool = _make_pool_with_models(4)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="m0"))
        for _ in range(6):
            pool.health_check("m0", success=False)
        affected = rms.handle_failover("m0")
        assert "Worker" in affected
        new_model = rms.get_binding("Worker").primary_model
        assert new_model != "m0"

    def test_failover_returns_empty_when_model_not_in_pool(self):
        pool = _make_pool_with_models(2)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        affected = rms.handle_failover("nonexistent")
        assert affected == []

    def test_failover_confidence_drops(self):
        pool = _make_pool_with_models(3)
        reg = _make_registry_with_profiles(pool)
        rms = RoleMergerSplitter(pool, reg)
        binding = RoleBinding(
            role="Worker", primary_model="m0", backup_models=["m1"], confidence=0.9
        )
        rms.set_binding(binding)
        for _ in range(6):
            pool.health_check("m0", success=False)
        rms.handle_failover("m0")
        assert rms.get_binding("Worker").confidence < 0.9

    def test_failover_no_replacement_available(self):
        pool = ModelPool()
        pool.register("only")
        reg = CapabilityRegistry()
        profile = CapabilityProfile(
            model_name="only", benchmark_score=0.9, historical_score=0.9,
            human_feedback_score=0.9, reliability=0.95,
        )
        reg.register(profile)
        rms = RoleMergerSplitter(pool, reg)
        rms.set_binding(RoleBinding(role="Worker", primary_model="only"))
        for _ in range(6):
            pool.health_check("only", success=False)
        affected = rms.handle_failover("only")
        assert "Worker" in affected
        # No replacement found, primary_model stays same (or unchanged)
