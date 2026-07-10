"""Engineering Simulation Agent unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainRegistry,
    DomainTask,
    DomainType,
    CalibrationStatus,
)
from dr_mma.engine.domains.engineering_sim import EngineeringSimAgent


class TestEngineeringSimAgentInit:
    def test_inherits_domain_agent(self):
        agent = EngineeringSimAgent()
        assert isinstance(agent, DomainAgent)

    def test_domain_type(self):
        agent = EngineeringSimAgent()
        assert agent.domain == DomainType.ENGINEERING_SIM

    def test_default_agent_id(self):
        agent = EngineeringSimAgent()
        assert agent.agent_id == "eng_sim_01"

    def test_custom_agent_id(self):
        agent = EngineeringSimAgent("my_sim")
        assert agent.agent_id == "my_sim"

    def test_profile_initialized_with_skills(self):
        agent = EngineeringSimAgent()
        assert len(agent.profile.skills) == 5


class TestGetDomainSkills:
    def test_returns_all_five_skills(self):
        agent = EngineeringSimAgent()
        skills = agent.get_domain_skills()
        expected = {
            "electromagnetic_field",
            "control_system",
            "dynamics_model",
            "numerical_method",
            "signal_processing",
        }
        assert set(skills.keys()) == expected

    def test_all_scores_are_floats(self):
        agent = EngineeringSimAgent()
        skills = agent.get_domain_skills()
        for score in skills.values():
            assert isinstance(score, float)

    def test_scores_in_valid_range(self):
        agent = EngineeringSimAgent()
        skills = agent.get_domain_skills()
        for name, score in skills.items():
            assert 0.0 <= score <= 1.0, f"{name}={score} out of [0,1]"


class TestGetCalibrationTasks:
    def test_at_least_three_tasks(self):
        agent = EngineeringSimAgent()
        tasks = agent.get_calibration_tasks()
        assert len(tasks) >= 3

    def test_task_has_required_fields(self):
        agent = EngineeringSimAgent()
        for task_def in agent.get_calibration_tasks():
            assert "name" in task_def
            assert "objective" in task_def
            assert "input" in task_def

    def test_all_subtask_types_covered(self):
        agent = EngineeringSimAgent()
        subtasks = {t["input"]["subtask"] for t in agent.get_calibration_tasks()}
        assert "electromagnetic_field" in subtasks
        assert "control_system" in subtasks
        assert "dynamics_model" in subtasks


class TestExecuteElectromagneticField:
    def _make_task(self, **kwargs):
        defaults = {
            "subtask": "electromagnetic_field",
            "voltage": 100.0,
            "geometry": "plate",
            "width": 0.01,
            "height": 0.001,
        }
        defaults.update(kwargs)
        return DomainTask(
            task_id="test_em",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Electromagnetic test",
            objective="Test EM field",
            input_data=defaults,
        )

    def test_plate_geometry(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(self._make_task())
        result = output["output_result"]
        assert result["field_strength"] > 0
        assert result["capacitance"] > 0
        assert result["energy_density"] >= 0

    def test_cylindrical_geometry(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(
            self._make_task(geometry="cyl", width=0.02, height=0.1)
        )
        result = output["output_result"]
        assert result["field_strength"] > 0
        assert result["capacitance"] > 0

    def test_output_structure(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(self._make_task())
        assert "task_id" in output
        assert "domain" in output
        assert "subtask" in output
        assert "input_params" in output
        assert "computation_summary" in output
        assert "output_result" in output
        assert "error_estimate" in output

    def test_error_estimate_present(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(self._make_task())
        assert isinstance(output["error_estimate"], (int, float))


class TestExecuteControlSystem:
    def _make_task(self, **kwargs):
        defaults = {
            "subtask": "control_system",
            "natural_freq": 10.0,
            "damping_ratio": 0.7,
            "input_type": "step",
            "amplitude": 1.0,
        }
        defaults.update(kwargs)
        return DomainTask(
            task_id="test_ctrl",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Control test",
            objective="Test control system",
            input_data=defaults,
        )

    def test_step_response(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(self._make_task(input_type="step"))
        result = output["output_result"]
        assert result["settling_time"] > 0
        assert 0 <= result["overshoot"] <= 100
        assert result["bandwidth"] > 0

    def test_underdamped_overshoot(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(
            self._make_task(damping_ratio=0.3)
        )
        result = output["output_result"]
        assert result["overshoot"] > 0

    def test_overdamped_no_overshoot(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(
            self._make_task(damping_ratio=1.5)
        )
        result = output["output_result"]
        assert result["overshoot"] == 0.0

    def test_ramp_steady_state_error(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(
            self._make_task(input_type="ramp", amplitude=2.0)
        )
        result = output["output_result"]
        assert result["steady_state_error"] > 0


class TestExecuteDynamicsModel:
    def _make_task(self, **kwargs):
        defaults = {
            "subtask": "dynamics_model",
            "mass": 1.0,
            "stiffness": 100.0,
            "damping": 1.0,
            "force_amplitude": 10.0,
            "force_freq": 1.0,
            "duration": 5.0,
        }
        defaults.update(kwargs)
        return DomainTask(
            task_id="test_dyn",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Dynamics test",
            objective="Test dynamics model",
            input_data=defaults,
        )

    def test_basic_response(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(self._make_task())
        result = output["output_result"]
        assert result["natural_freq_hz"] > 0
        assert result["max_displacement"] > 0
        assert result["resonance_ratio"] > 0

    def test_resonance_near_natural_freq(self):
        agent = EngineeringSimAgent()
        # m=1, k=100 => fn = sqrt(100)/2pi ~ 1.59 Hz
        output = agent.execute_domain_task(
            self._make_task(force_freq=1.59)
        )
        result = output["output_result"]
        # Near resonance, amplification should be significant
        assert result["resonance_ratio"] > 1.0

    def test_displacement_samples(self):
        agent = EngineeringSimAgent()
        output = agent.execute_domain_task(
            self._make_task(duration=2.0)
        )
        result = output["output_result"]
        assert result["displacement_samples_count"] > 10


class TestUnsupportedSubtask:
    def test_unknown_subtask_returns_error(self):
        agent = EngineeringSimAgent()
        task = DomainTask(
            task_id="bad",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Bad task",
            objective="Test error",
            input_data={"subtask": "unknown_type"},
        )
        output = agent.execute_domain_task(task)
        assert "error" in output
        assert "unknown_type" in output["error"]


class TestValidateOutput:
    def test_valid_em_output_passes(self):
        agent = EngineeringSimAgent()
        task = DomainTask(
            task_id="v1",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="EM validate",
            objective="Validate EM",
            input_data={
                "subtask": "electromagnetic_field",
                "voltage": 100.0,
                "geometry": "plate",
            },
        )
        output = agent.execute_domain_task(task)
        passed, issues = agent.validate_output(output)
        assert passed is True
        assert len(issues) == 0

    def test_valid_control_output_passes(self):
        agent = EngineeringSimAgent()
        task = DomainTask(
            task_id="v2",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Control validate",
            objective="Validate control",
            input_data={
                "subtask": "control_system",
                "natural_freq": 10.0,
                "damping_ratio": 0.7,
            },
        )
        output = agent.execute_domain_task(task)
        passed, issues = agent.validate_output(output)
        assert passed is True

    def test_valid_dynamics_output_passes(self):
        agent = EngineeringSimAgent()
        task = DomainTask(
            task_id="v3",
            domain_type=DomainType.ENGINEERING_SIM,
            task_name="Dynamics validate",
            objective="Validate dynamics",
            input_data={
                "subtask": "dynamics_model",
                "mass": 1.0,
                "stiffness": 100.0,
            },
        )
        output = agent.execute_domain_task(task)
        passed, issues = agent.validate_output(output)
        assert passed is True

    def test_missing_keys_detected(self):
        agent = EngineeringSimAgent()
        passed, issues = agent.validate_output({})
        assert passed is False
        assert any("Missing" in i for i in issues)

    def test_negative_field_strength_detected(self):
        agent = EngineeringSimAgent()
        output = {
            "task_id": "x",
            "domain": "engineering_simulation",
            "subtask": "electromagnetic_field",
            "output_result": {"field_strength": -1.0, "capacitance": 1e-10},
            "error_estimate": 0.0,
        }
        passed, issues = agent.validate_output(output)
        assert passed is False
        assert any("non-negative" in i for i in issues)

    def test_nan_detected(self):
        import math as m
        agent = EngineeringSimAgent()
        output = {
            "task_id": "x",
            "domain": "engineering_simulation",
            "subtask": "control_system",
            "output_result": {"settling_time": float("nan")},
            "error_estimate": 0.0,
        }
        passed, issues = agent.validate_output(output)
        assert passed is False
        assert any("NaN" in i for i in issues)


class TestCalibration:
    def test_full_calibration_runs(self):
        agent = EngineeringSimAgent()
        results = agent.calibrate()
        assert len(results) >= 3
        assert all(r.passed for r in results)

    def test_calibration_updates_profile(self):
        agent = EngineeringSimAgent()
        agent.calibrate()
        assert agent.profile.sample_count > 0
        assert agent.profile.calibration_status == CalibrationStatus.CALIBRATED

    def test_calibration_summary(self):
        agent = EngineeringSimAgent()
        agent.calibrate()
        summary = agent.calibration_summary()
        assert summary["total_runs"] >= 3
        assert summary["avg_score"] > 0.5


class TestDomainRegistryIntegration:
    def test_register_and_query(self):
        reg = DomainRegistry()
        agent = EngineeringSimAgent("sim_01")
        reg.register(agent)
        agents = reg.get_agents_by_domain(DomainType.ENGINEERING_SIM)
        assert len(agents) == 1
        assert agents[0].agent_id == "sim_01"

    def test_find_best_agent(self):
        reg = DomainRegistry()
        agent = EngineeringSimAgent("sim_01")
        reg.register(agent)
        best = reg.find_best_agent("electromagnetic_field")
        assert best is not None
        assert best.agent_id == "sim_01"
