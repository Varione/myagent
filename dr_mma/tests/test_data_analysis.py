"""DataAnalysisAgent unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainRegistry,
    DomainType,
    CalibrationStatus,
    DomainTask,
)
from dr_mma.engine.domains.data_analysis import (
    DataAnalysisAgent,
    _compute_mean,
    _compute_median,
    _compute_std_dev,
    _compute_quartile,
    _compute_descriptive_stats,
    _t_test_independent,
    _generate_report,
)


@pytest.fixture
def da_agent():
    return DataAnalysisAgent()


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestDataAnalysisAgentInit:
    def test_inherits_domain_agent(self, da_agent):
        assert isinstance(da_agent, DomainAgent)

    def test_domain_is_data_analysis(self, da_agent):
        assert da_agent.domain == DomainType.DATA_ANALYSIS

    def test_default_agent_id(self, da_agent):
        assert da_agent.agent_id == "data_analysis_1"

    def test_custom_agent_id(self):
        agent = DataAnalysisAgent(agent_id="my_da")
        assert agent.agent_id == "my_da"

    def test_profile_initialized_with_skills(self, da_agent):
        skills = da_agent.get_domain_skills()
        for skill_name in skills:
            assert skill_name in da_agent.profile.skills


# ---------------------------------------------------------------------------
# get_domain_skills tests
# ---------------------------------------------------------------------------

class TestGetDomainSkills:
    def test_returns_all_five_skills(self, da_agent):
        skills = da_agent.get_domain_skills()
        expected_keys = {
            "statistical_modeling",
            "data_visualization",
            "report_generation",
            "hypothesis_testing",
            "descriptive_stats",
        }
        assert set(skills.keys()) == expected_keys

    def test_skill_scores_match_spec(self, da_agent):
        skills = da_agent.get_domain_skills()
        assert skills["statistical_modeling"] == 0.8
        assert skills["data_visualization"] == 0.75
        assert skills["report_generation"] == 0.85
        assert skills["hypothesis_testing"] == 0.7
        assert skills["descriptive_stats"] == 0.9

    def test_all_scores_are_floats(self, da_agent):
        skills = da_agent.get_domain_skills()
        for score in skills.values():
            assert isinstance(score, float)

    def test_all_scores_between_0_and_1(self, da_agent):
        skills = da_agent.get_domain_skills()
        for score in skills.values():
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# get_calibration_tasks tests
# ---------------------------------------------------------------------------

class TestGetCalibrationTasks:
    def test_at_least_three_tasks(self, da_agent):
        tasks = da_agent.get_calibration_tasks()
        assert len(tasks) >= 3

    def test_task_has_required_fields(self, da_agent):
        tasks = da_agent.get_calibration_tasks()
        for task in tasks:
            assert "name" in task
            assert "objective" in task
            assert "input" in task

    def test_covers_descriptive_stats(self, da_agent):
        tasks = da_agent.get_calibration_tasks()
        subtasks = {t["input"]["subtask"] for t in tasks}
        assert "descriptive_stats" in subtasks

    def test_covers_hypothesis_testing(self, da_agent):
        tasks = da_agent.get_calibration_tasks()
        subtasks = {t["input"]["subtask"] for t in tasks}
        assert "hypothesis_testing" in subtasks

    def test_covers_report_generation(self, da_agent):
        tasks = da_agent.get_calibration_tasks()
        subtasks = {t["input"]["subtask"] for t in tasks}
        assert "report_generation" in subtasks


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestComputeMean:
    def test_simple_mean(self):
        assert _compute_mean([2, 4, 6]) == 4.0

    def test_float_mean(self):
        assert abs(_compute_mean([1.5, 2.5, 3.0]) - 2.333333) < 1e-4

    def test_single_element(self):
        assert _compute_mean([42]) == 42.0

    def test_empty_list(self):
        assert _compute_mean([]) == 0.0


class TestComputeMedian:
    def test_odd_count(self):
        assert _compute_median([3, 1, 2]) == 2.0

    def test_even_count(self):
        assert _compute_median([1, 2, 3, 4]) == 2.5

    def test_single_element(self):
        assert _compute_median([7]) == 7.0

    def test_empty_list(self):
        assert _compute_median([]) == 0.0


class TestComputeStdDev:
    def test_known_std_dev(self):
        # Population std dev of [2,4,4,4,5,5,7,9] is exactly 2.0
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        result = _compute_std_dev(data, population=True)
        assert abs(result - 2.0) < 1e-4

    def test_single_element(self):
        assert _compute_std_dev([42]) == 0.0

    def test_empty_list(self):
        assert _compute_std_dev([]) == 0.0

    def test_population_flag(self):
        data = [10, 12, 14]
        sample_sd = _compute_std_dev(data, population=False)
        pop_sd = _compute_std_dev(data, population=True)
        assert pop_sd < sample_sd


class TestComputeQuartile:
    def test_q1_of_five(self):
        data = sorted([2, 4, 6, 8, 10])
        q1 = _compute_quartile(data, 0.25)
        assert abs(q1 - 4.0) < 1e-9

    def test_q3_of_five(self):
        data = sorted([2, 4, 6, 8, 10])
        q3 = _compute_quartile(data, 0.75)
        assert abs(q3 - 8.0) < 1e-9

    def test_single_element(self):
        assert _compute_quartile([42], 0.5) == 42.0

    def test_empty(self):
        assert _compute_quartile([], 0.5) == 0.0


class TestComputeDescriptiveStats:
    def test_known_data(self):
        data = [2, 4, 6, 8, 10]
        result = _compute_descriptive_stats(data)
        assert result["mean"] == 6.0
        assert result["median"] == 6.0
        assert result["min"] == 2
        assert result["max"] == 10
        assert result["count"] == 5

    def test_empty_data(self):
        result = _compute_descriptive_stats([])
        assert result["count"] == 0
        assert result["mean"] == 0.0
        assert result["min"] is None
        assert result["max"] is None

    def test_single_element(self):
        result = _compute_descriptive_stats([42])
        assert result["mean"] == 42.0
        assert result["median"] == 42.0
        assert result["std_dev"] == 0.0
        assert result["min"] == 42
        assert result["max"] == 42
        assert result["count"] == 1


class TestTTestIndependent:
    def test_clearly_different_groups(self):
        # Group A is much higher than Group B
        group_a = [100, 102, 98, 101, 99]
        group_b = [50, 52, 48, 51, 49]
        result = _t_test_independent(group_a, group_b)
        assert result["significant"] is True
        assert abs(result["t_statistic"]) > 0

    def test_identical_groups(self):
        group_a = [1, 2, 3, 4, 5]
        group_b = [1, 2, 3, 4, 5]
        result = _t_test_independent(group_a, group_b)
        assert result["significant"] is False
        assert abs(result["t_statistic"]) < 1e-9

    def test_insufficient_data(self):
        result = _t_test_independent([1], [2])
        assert result["significant"] is False
        assert "Insufficient" in result["conclusion"]

    def test_result_has_all_keys(self):
        result = _t_test_independent([1, 2, 3], [4, 5, 6])
        assert "t_statistic" in result
        assert "df" in result
        assert "significant" in result
        assert "conclusion" in result
        assert isinstance(result["significant"], bool)


class TestGenerateReport:
    def test_basic_report(self):
        stats = {
            "mean": 6.0,
            "median": 6.0,
            "std_dev": 2.83,
            "min": 2,
            "max": 10,
            "q1": 4.0,
            "q3": 8.0,
            "count": 5,
        }
        result = _generate_report(stats)
        assert "report_text" in result
        assert "sections" in result
        assert len(result["report_text"]) > 0
        assert "Overview" in result["sections"]

    def test_empty_stats(self):
        result = _generate_report({})
        assert "report_text" in result
        assert isinstance(result["report_text"], str)


# ---------------------------------------------------------------------------
# execute_domain_task tests
# ---------------------------------------------------------------------------

class TestExecuteDescriptiveStats:
    def test_normal_data(self, da_agent):
        task = DomainTask(
            task_id="ds_1",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Compute stats",
            objective="Compute descriptive statistics",
            input_data={
                "subtask": "descriptive_stats",
                "data": [2, 4, 6, 8, 10],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert result["subtask"] == "descriptive_stats"
        assert result["mean"] == 6.0
        assert result["median"] == 6.0
        assert result["count"] == 5
        assert result["task_id"] == "ds_1"

    def test_empty_data(self, da_agent):
        task = DomainTask(
            task_id="ds_2",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Empty data",
            objective="Handle empty data",
            input_data={
                "subtask": "descriptive_stats",
                "data": [],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert result["count"] == 0

    def test_single_element(self, da_agent):
        task = DomainTask(
            task_id="ds_3",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Single element",
            objective="Handle single element",
            input_data={
                "subtask": "descriptive_stats",
                "data": [42],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert result["mean"] == 42.0
        assert result["std_dev"] == 0.0

    def test_non_list_data(self, da_agent):
        task = DomainTask(
            task_id="ds_4",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Invalid data type",
            objective="Handle non-list input",
            input_data={
                "subtask": "descriptive_stats",
                "data": "not a list",
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "error"

    def test_non_numeric_element(self, da_agent):
        task = DomainTask(
            task_id="ds_5",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Non-numeric element",
            objective="Handle non-numeric data",
            input_data={
                "subtask": "descriptive_stats",
                "data": [1, 2, "three"],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "error"

    def test_negative_numbers(self, da_agent):
        task = DomainTask(
            task_id="ds_6",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Negative numbers",
            objective="Handle negative values",
            input_data={
                "subtask": "descriptive_stats",
                "data": [-5, -3, -1, 0, 2],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert result["min"] == -5
        assert result["max"] == 2


class TestExecuteHypothesisTesting:
    def test_significant_difference(self, da_agent):
        task = DomainTask(
            task_id="ht_1",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Significant diff",
            objective="Detect significant difference",
            input_data={
                "subtask": "hypothesis_testing",
                "group_a": [100, 102, 98, 101],
                "group_b": [50, 52, 48, 51],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert result["significant"] is True
        assert abs(result["t_statistic"]) > 0

    def test_no_significant_difference(self, da_agent):
        task = DomainTask(
            task_id="ht_2",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="No significant diff",
            objective="Detect no significant difference",
            input_data={
                "subtask": "hypothesis_testing",
                "group_a": [10, 12, 14],
                "group_b": [11, 13, 15],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        # Groups are very similar
        assert isinstance(result["significant"], bool)

    def test_empty_groups(self, da_agent):
        task = DomainTask(
            task_id="ht_3",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Empty groups",
            objective="Handle empty groups",
            input_data={
                "subtask": "hypothesis_testing",
                "group_a": [],
                "group_b": [1, 2],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "error"

    def test_result_has_all_keys(self, da_agent):
        task = DomainTask(
            task_id="ht_4",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Full output",
            objective="Check all keys present",
            input_data={
                "subtask": "hypothesis_testing",
                "group_a": [1, 2, 3],
                "group_b": [4, 5, 6],
            },
        )
        result = da_agent.execute_domain_task(task)
        assert "t_statistic" in result
        assert "df" in result
        assert "significant" in result
        assert "conclusion" in result


class TestExecuteReportGeneration:
    def test_normal_report(self, da_agent):
        task = DomainTask(
            task_id="rg_1",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Generate report",
            objective="Create analysis report",
            input_data={
                "subtask": "report_generation",
                "stats": {
                    "mean": 6.0,
                    "median": 6.0,
                    "std_dev": 2.83,
                    "min": 2,
                    "max": 10,
                    "q1": 4.0,
                    "q3": 8.0,
                    "count": 5,
                },
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert len(result["report_text"]) > 0
        assert len(result["sections"]) >= 3

    def test_empty_stats(self, da_agent):
        task = DomainTask(
            task_id="rg_2",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Empty stats",
            objective="Handle empty stats dict",
            input_data={
                "subtask": "report_generation",
                "stats": {},
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "completed"
        assert isinstance(result["report_text"], str)

    def test_invalid_stats_type(self, da_agent):
        task = DomainTask(
            task_id="rg_3",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Invalid stats type",
            objective="Handle non-dict stats",
            input_data={
                "subtask": "report_generation",
                "stats": "not a dict",
            },
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "error"


class TestUnknownSubtask:
    def test_unknown_subtask_returns_error(self, da_agent):
        task = DomainTask(
            task_id="unknown_1",
            domain_type=DomainType.DATA_ANALYSIS,
            task_name="Unknown subtask",
            objective="Test unknown type",
            input_data={"subtask": "unknown_type"},
        )
        result = da_agent.execute_domain_task(task)
        assert result["status"] == "error"
        assert "Unknown subtask" in result["message"]


# ---------------------------------------------------------------------------
# validate_output tests
# ---------------------------------------------------------------------------

class TestValidateOutput:
    def test_valid_descriptive_stats(self, da_agent):
        output = {
            "task_id": "v1",
            "subtask": "descriptive_stats",
            "mean": 6.0,
            "median": 6.0,
            "std_dev": 2.83,
            "min": 2,
            "max": 10,
            "q1": 4.0,
            "q3": 8.0,
            "count": 5,
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is True
        assert len(issues) == 0

    def test_missing_descriptive_stats_key(self, da_agent):
        output = {
            "task_id": "v2",
            "subtask": "descriptive_stats",
            "mean": 6.0,
            # missing other keys
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is False
        assert len(issues) > 0

    def test_valid_hypothesis_testing(self, da_agent):
        output = {
            "task_id": "v3",
            "subtask": "hypothesis_testing",
            "t_statistic": 2.5,
            "df": 8,
            "significant": True,
            "conclusion": "Reject null hypothesis",
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is True

    def test_significant_not_bool(self, da_agent):
        output = {
            "task_id": "v4",
            "subtask": "hypothesis_testing",
            "t_statistic": 2.5,
            "df": 8,
            "significant": "yes",
            "conclusion": "Reject null hypothesis",
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is False
        assert any("boolean" in i for i in issues)

    def test_valid_report_generation(self, da_agent):
        output = {
            "task_id": "v5",
            "subtask": "report_generation",
            "report_text": "Some report text",
            "sections": ["Overview"],
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is True

    def test_missing_task_id(self, da_agent):
        output = {"subtask": "descriptive_stats", "mean": 1.0}
        passed, issues = da_agent.validate_output(output)
        assert passed is False
        assert any("task_id" in i for i in issues)

    def test_non_dict_output(self, da_agent):
        passed, issues = da_agent.validate_output("not a dict")
        assert passed is False

    def test_report_text_not_string(self, da_agent):
        output = {
            "task_id": "v6",
            "subtask": "report_generation",
            "report_text": 12345,
            "sections": [],
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is False

    def test_sections_not_list(self, da_agent):
        output = {
            "task_id": "v7",
            "subtask": "report_generation",
            "report_text": "text",
            "sections": "not a list",
        }
        passed, issues = da_agent.validate_output(output)
        assert passed is False


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_runs(self, da_agent):
        results = da_agent.calibrate()
        assert len(results) >= 3

    def test_calibration_updates_profile(self, da_agent):
        da_agent.calibrate()
        assert da_agent.profile.sample_count > 0

    def test_calibration_summary_available(self, da_agent):
        da_agent.calibrate()
        summary = da_agent.calibration_summary()
        assert summary["total_runs"] >= 3

    def test_calibration_results_have_details(self, da_agent):
        results = da_agent.calibrate()
        for r in results:
            assert "issues" in r.details

    def test_calibration_status_after_run(self, da_agent):
        da_agent.calibrate()
        # All calibration tasks should pass, so status should be CALIBRATED or DEGRADED
        assert da_agent.profile.calibration_status in (
            CalibrationStatus.CALIBRATED,
            CalibrationStatus.DEGRADED,
        )


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_register_data_analysis_agent(self):
        agent = DataAnalysisAgent(agent_id="da_1")
        registry = DomainRegistry()
        registry.register(agent)

        assert registry.agent_count == 1
        found = registry.get_agent("da_1")
        assert found is not None
        assert found.domain == DomainType.DATA_ANALYSIS

    def test_find_by_domain(self):
        agent = DataAnalysisAgent(agent_id="da_1")
        registry = DomainRegistry()
        registry.register(agent)

        agents = registry.get_agents_by_domain(DomainType.DATA_ANALYSIS)
        assert len(agents) == 1

    def test_find_best_descriptive_stats(self):
        agent = DataAnalysisAgent(agent_id="da_1")
        agent.profile.skills["descriptive_stats"] = 0.9
        registry = DomainRegistry()
        registry.register(agent)

        best = registry.find_best_agent("descriptive_stats")
        assert best is not None
        assert best.agent_id == "da_1"
