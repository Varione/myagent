"""Domain Agent Base unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainRegistry,
    DomainType,
    CapabilityProfile,
    CalibrationStatus,
    DomainTask,
)


class MockDomainAgent(DomainAgent):
    """测试用模拟 Agent。"""

    def __init__(self, agent_id: str = "mock_1"):
        super().__init__(agent_id, DomainType.DATA_ANALYSIS)
        self.profile.skills = {"test_skill": 0.8}

    def get_domain_skills(self):
        return {"test_skill": 0.8, "another_skill": 0.6}

    def get_calibration_tasks(self):
        return [
            {"name": "calib_1", "objective": "Test task 1", "input": {}},
            {"name": "calib_2", "objective": "Test task 2", "input": {}},
        ]

    def execute_domain_task(self, task):
        return {"result": "ok", "task_id": task.task_id}

    def validate_output(self, output):
        return True, []


class TestCapabilityProfile:
    def test_to_dict(self):
        p = CapabilityProfile(domain=DomainType.DATA_ANALYSIS)
        d = p.to_dict()
        assert d["domain"] == "data_analysis"

    def test_average_skill_score(self):
        p = CapabilityProfile(
            domain=DomainType.DATA_ANALYSIS,
            skills={"a": 0.8, "b": 0.6},
        )
        assert abs(p.average_skill_score() - 0.7) < 1e-9

    def test_empty_skills(self):
        p = CapabilityProfile(domain=DomainType.DATA_ANALYSIS)
        assert p.average_skill_score() == 0.0


class TestDomainAgent:
    def test_initial_profile(self):
        agent = MockDomainAgent()
        assert agent.domain == DomainType.DATA_ANALYSIS
        assert agent.profile.domain == DomainType.DATA_ANALYSIS

    def test_calibration(self):
        agent = MockDomainAgent()
        results = agent.calibrate()
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_calibration_updates_profile(self):
        agent = MockDomainAgent()
        agent.calibrate()
        assert agent.profile.sample_count > 0
        assert agent.profile.calibration_status == CalibrationStatus.CALIBRATED

    def test_calibration_summary(self):
        agent = MockDomainAgent()
        agent.calibrate()
        s = agent.calibration_summary()
        assert s["total_runs"] == 2
        assert s["avg_score"] > 0


class TestDomainRegistry:
    def _setup_registry(self):
        reg = DomainRegistry()
        a1 = MockDomainAgent("mock_1")
        a2 = MockDomainAgent("mock_2")
        a2.domain = DomainType.CODE_DEVELOPMENT
        a2.profile.domain = DomainType.CODE_DEVELOPMENT
        a2.profile.skills = {"python_coding": 0.9}
        reg.register(a1)
        reg.register(a2)
        return reg

    def test_register_and_count(self):
        reg = self._setup_registry()
        assert reg.agent_count == 2

    def test_unregister(self):
        reg = self._setup_registry()
        assert reg.unregister("mock_1") is True
        assert reg.agent_count == 1

    def test_get_by_domain(self):
        reg = self._setup_registry()
        agents = reg.get_agents_by_domain(DomainType.DATA_ANALYSIS)
        assert len(agents) == 1

    def test_find_best_agent(self):
        reg = self._setup_registry()
        best = reg.find_best_agent("python_coding")
        assert best is not None
        assert best.agent_id == "mock_2"

    def test_calibrate_all(self):
        reg = self._setup_registry()
        results = reg.calibrate_all()
        assert len(results) == 2

    def test_registry_summary(self):
        reg = self._setup_registry()
        s = reg.registry_summary()
        assert s["total_agents"] == 2
        assert "data_analysis" in s["by_domain"]


class TestDomainTask:
    def test_task_creation(self):
        t = DomainTask(
            task_id="T1",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Test",
            objective="Test objective",
        )
        assert t.task_id == "T1"
        assert t.domain_type == DomainType.DATA_ANALYSIS
