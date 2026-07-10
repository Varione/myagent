"""CodeDevAgent unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainRegistry,
    DomainType,
    CalibrationStatus,
    DomainTask,
)
from dr_mma.engine.domains.code_dev import (
    CodeDevAgent,
    _check_naming_convention,
    _count_complexity,
    _check_syntax,
    _analyze_code_quality,
)


@pytest.fixture
def code_agent():
    return CodeDevAgent()


class TestCodeDevAgentInit:
    def test_inherits_domain_agent(self, code_agent):
        assert isinstance(code_agent, DomainAgent)

    def test_domain_is_code_development(self, code_agent):
        assert code_agent.domain == DomainType.CODE_DEVELOPMENT

    def test_default_agent_id(self, code_agent):
        assert code_agent.agent_id == "code_dev_1"

    def test_custom_agent_id(self):
        agent = CodeDevAgent(agent_id="my_code_agent")
        assert agent.agent_id == "my_code_agent"

    def test_profile_initialized_with_skills(self, code_agent):
        skills = code_agent.get_domain_skills()
        for skill_name in skills:
            assert skill_name in code_agent.profile.skills


class TestGetDomainSkills:
    def test_returns_all_five_skills(self, code_agent):
        skills = code_agent.get_domain_skills()
        expected = {
            "python_coding",
            "code_review",
            "test_generation",
            "debug_analysis",
            "architecture_design",
        }
        assert set(skills.keys()) == expected

    def test_skill_scores_are_floats(self, code_agent):
        skills = code_agent.get_domain_skills()
        for score in skills.values():
            assert isinstance(score, float)

    def test_initial_scores_are_positive(self, code_agent):
        skills = code_agent.get_domain_skills()
        for score in skills.values():
            assert score > 0


class TestGetCalibrationTasks:
    def test_at_least_three_tasks(self, code_agent):
        tasks = code_agent.get_calibration_tasks()
        assert len(tasks) >= 3

    def test_task_has_required_fields(self, code_agent):
        tasks = code_agent.get_calibration_tasks()
        for task in tasks:
            assert "name" in task
            assert "objective" in task
            assert "input" in task

    def test_covers_all_subtask_types(self, code_agent):
        tasks = code_agent.get_calibration_tasks()
        types = {t["input"]["subtask_type"] for t in tasks}
        assert "code_review" in types
        assert "code_generation" in types
        assert "test_generation" in types


class TestCodeGeneration:
    def test_add_description(self, code_agent):
        task = DomainTask(
            task_id="gen_1",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate add function",
            objective="Create an add function",
            input_data={
                "subtask_type": "code_generation",
                "description": "A function that adds two numbers",
                "language": "python",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "completed"
        assert result["subtask_type"] == "code_generation"
        assert "add(" in result["generated_code"]
        assert result["task_id"] == "gen_1"

    def test_fibonacci_description(self, code_agent):
        task = DomainTask(
            task_id="gen_2",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate fibonacci",
            objective="Create fibonacci function",
            input_data={
                "subtask_type": "code_generation",
                "description": "A fibonacci sequence calculator",
                "language": "python",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert "fibonacci" in result["generated_code"]
        assert result["status"] == "completed"

    def test_sort_description(self, code_agent):
        task = DomainTask(
            task_id="gen_3",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate sort",
            objective="Create sort function",
            input_data={
                "subtask_type": "code_generation",
                "description": "A sort algorithm",
                "language": "python",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert "sort" in result["generated_code"].lower()

    def test_unknown_description_generates_template(self, code_agent):
        task = DomainTask(
            task_id="gen_4",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate unknown",
            objective="Unknown function",
            input_data={
                "subtask_type": "code_generation",
                "description": "Calculate something weird",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "completed"
        assert "def " in result["generated_code"]

    def test_generated_code_is_valid_syntax(self, code_agent):
        task = DomainTask(
            task_id="gen_5",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate valid code",
            objective="Must be valid Python",
            input_data={
                "subtask_type": "code_generation",
                "description": "A function that adds two numbers",
            },
        )
        result = code_agent.execute_domain_task(task)
        syntax_ok, issues = _check_syntax(result["generated_code"])
        assert syntax_ok is True

    def test_result_has_self_analysis(self, code_agent):
        task = DomainTask(
            task_id="gen_6",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="With analysis",
            objective="Should include self analysis",
            input_data={
                "subtask_type": "code_generation",
                "description": "A function that adds two numbers",
            },
        )
        result = code_agent.execute_domain_task(task)
        assert "self_analysis" in result
        assert isinstance(result["self_analysis"], dict)


class TestCodeReview:
    def test_valid_code_passes(self, code_agent):
        task = DomainTask(
            task_id="review_1",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review valid code",
            objective="Check good code",
            input_data={
                "subtask_type": "code_review",
                "code": (
                    "def my_func(x):\n"
                    '    """Double the input."""\n'
                    "    return x * 2\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "completed"
        assert result["analysis"]["syntax_valid"] is True
        assert result["quality_score"] > 0

    def test_syntax_error_detected(self, code_agent):
        task = DomainTask(
            task_id="review_2",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review bad syntax",
            objective="Detect syntax error",
            input_data={
                "subtask_type": "code_review",
                "code": "def broken(",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["analysis"]["syntax_valid"] is False
        assert len(result["analysis"]["issues"]) > 0

    def test_complexity_detected(self, code_agent):
        task = DomainTask(
            task_id="review_3",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review complex code",
            objective="Detect high complexity",
            input_data={
                "subtask_type": "code_review",
                "code": (
                    "def complex_func(x):\n"
                    "    if x > 0:\n"
                    "        if x > 10:\n"
                    "            if x > 20:\n"
                    "                if x > 30:\n"
                    "                    if x > 40:\n"
                    "                        if x > 50:\n"
                    "                            if x > 60:\n"
                    "                                if x > 70:\n"
                    "                                    if x > 80:\n"
                    "                                        if x > 90:\n"
                    "                                            if x > 100:\n"
                    "                                                if x > 110:\n"
                    "                                                    return 1\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        complexity_issues = [
            i for i in result["analysis"]["issues"]
            if "cyclomatic complexity" in i
        ]
        assert len(complexity_issues) > 0

    def test_missing_docstring_detected(self, code_agent):
        task = DomainTask(
            task_id="review_4",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review no docstring",
            objective="Detect missing docstring",
            input_data={
                "subtask_type": "code_review",
                "code": (
                    "def no_doc(x):\n"
                    "    return x * 2\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        doc_issues = [
            i for i in result["analysis"]["issues"]
            if "missing docstring" in i.lower()
        ]
        assert len(doc_issues) > 0

    def test_naming_violation_detected(self, code_agent):
        task = DomainTask(
            task_id="review_5",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review bad naming",
            objective="Detect naming issues",
            input_data={
                "subtask_type": "code_review",
                "code": (
                    'class badClassName:\n'
                    '    """Bad class name."""\n'
                    "    pass\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        naming_issues = [
            i for i in result["analysis"]["issues"]
            if "should follow" in i or "should be" in i
        ]
        assert len(naming_issues) > 0

    def test_empty_code_returns_error(self, code_agent):
        task = DomainTask(
            task_id="review_6",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Review empty",
            objective="Empty code handling",
            input_data={
                "subtask_type": "code_review",
                "code": "",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "error"

    def test_result_has_recommendations(self, code_agent):
        task = DomainTask(
            task_id="review_7",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="With recommendations",
            objective="Should have recommendations",
            input_data={
                "subtask_type": "code_review",
                "code": (
                    "def no_doc(x):\n"
                    "    return x * 2\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)


class TestTestGeneration:
    def test_generates_valid_test_code(self, code_agent):
        task = DomainTask(
            task_id="test_1",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Generate tests",
            objective="Create unit tests",
            input_data={
                "subtask_type": "test_generation",
                "code": (
                    "def add(a, b):\n"
                    '    """Add two numbers."""\n'
                    "    return a + b\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "completed"
        assert "unittest" in result["generated_tests"]
        assert "test_add" in result["generated_tests"]

    def test_generated_tests_are_valid_syntax(self, code_agent):
        task = DomainTask(
            task_id="test_2",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Valid test syntax",
            objective="Tests must be valid Python",
            input_data={
                "subtask_type": "test_generation",
                "code": (
                    "def multiply(x, y):\n"
                    '    """Multiply two numbers."""\n'
                    "    return x * y\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        syntax_ok, issues = _check_syntax(result["generated_tests"])
        assert syntax_ok is True

    def test_empty_code_returns_error(self, code_agent):
        task = DomainTask(
            task_id="test_3",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Empty source",
            objective="Handle empty input",
            input_data={
                "subtask_type": "test_generation",
                "code": "",
            },
        )
        result = code_agent.execute_domain_task(task)

        assert result["status"] == "error"

    def test_multiple_functions_generate_tests(self, code_agent):
        task = DomainTask(
            task_id="test_4",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Multiple functions",
            objective="Generate tests for multiple functions",
            input_data={
                "subtask_type": "test_generation",
                "code": (
                    "def add(a, b):\n"
                    '    """Add."""\n'
                    "    return a + b\n\n"
                    "def sub(a, b):\n"
                    '    """Subtract."""\n'
                    "    return a - b\n"
                ),
            },
        )
        result = code_agent.execute_domain_task(task)

        assert "test_add" in result["generated_tests"]
        assert "test_sub" in result["generated_tests"]


class TestValidateOutput:
    def test_valid_generation_output(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "code_generation",
            "task_id": "t1",
            "generated_code": "def add(a, b):\n    return a + b\n",
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is True

    def test_valid_review_output(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "code_review",
            "task_id": "t2",
            "analysis": {"syntax_valid": True, "issues": [], "metrics": {}},
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is True

    def test_valid_test_output(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "test_generation",
            "task_id": "t3",
            "generated_tests": "import unittest\n\nclass Test(unittest.TestCase):\n    pass\n",
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is True

    def test_missing_status_fails(self, code_agent):
        output = {"task_id": "t4"}
        passed, issues = code_agent.validate_output(output)
        assert passed is False
        assert any("status" in i for i in issues)

    def test_missing_task_id_fails(self, code_agent):
        output = {"status": "completed"}
        passed, issues = code_agent.validate_output(output)
        assert passed is False
        assert any("task_id" in i for i in issues)

    def test_empty_generated_code_fails(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "code_generation",
            "task_id": "t5",
            "generated_code": "",
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is False

    def test_invalid_syntax_in_generated_code_fails(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "code_generation",
            "task_id": "t6",
            "generated_code": "def broken(",
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is False

    def test_non_dict_output_fails(self, code_agent):
        passed, issues = code_agent.validate_output("not a dict")
        assert passed is False

    def test_missing_analysis_fields_fails(self, code_agent):
        output = {
            "status": "completed",
            "subtask_type": "code_review",
            "task_id": "t7",
            "analysis": {},
        }
        passed, issues = code_agent.validate_output(output)
        assert passed is False


class TestUnknownSubtask:
    def test_unknown_subtask_returns_error(self, code_agent):
        task = DomainTask(
            task_id="unknown_1",
            domain_type=DomainType.CODE_DEVELOPMENT,
            task_name="Unknown",
            objective="Test unknown type",
            input_data={
                "subtask_type": "unknown_type",
            },
        )
        result = code_agent.execute_domain_task(task)
        assert result["status"] == "error"


class TestCalibration:
    def test_calibration_runs(self, code_agent):
        results = code_agent.calibrate()
        assert len(results) >= 3

    def test_calibration_updates_profile(self, code_agent):
        code_agent.calibrate()
        assert code_agent.profile.sample_count > 0

    def test_calibration_summary_available(self, code_agent):
        code_agent.calibrate()
        summary = code_agent.calibration_summary()
        assert summary["total_runs"] >= 3


class TestRegistryIntegration:
    def test_register_code_dev_agent(self):
        agent = CodeDevAgent(agent_id="dev_1")
        registry = DomainRegistry()
        registry.register(agent)

        assert registry.agent_count == 1
        found = registry.get_agent("dev_1")
        assert found is not None
        assert found.domain == DomainType.CODE_DEVELOPMENT

    def test_find_by_domain(self):
        agent = CodeDevAgent(agent_id="dev_1")
        registry = DomainRegistry()
        registry.register(agent)

        agents = registry.get_agents_by_domain(DomainType.CODE_DEVELOPMENT)
        assert len(agents) == 1

    def test_find_best_python_coding(self):
        agent = CodeDevAgent(agent_id="dev_1")
        agent.profile.skills["python_coding"] = 0.9
        registry = DomainRegistry()
        registry.register(agent)

        best = registry.find_best_agent("python_coding")
        assert best is not None
        assert best.agent_id == "dev_1"


class TestHelperFunctions:
    def test_check_naming_valid_class(self):
        issues = _check_naming_convention("MyClass", "class")
        assert len(issues) == 0

    def test_check_naming_invalid_class(self):
        issues = _check_naming_convention("my_class", "class")
        assert len(issues) > 0

    def test_check_naming_valid_function(self):
        issues = _check_naming_convention("my_function", "function")
        assert len(issues) == 0

    def test_check_naming_keyword(self):
        issues = _check_naming_convention("class", "function")
        assert len(issues) > 0

    def test_count_complexity_simple(self):
        import ast as _ast
        tree = _ast.parse("def f(): return 1")
        func = tree.body[0]
        cc = _count_complexity(func)
        assert cc == 1

    def test_count_complexity_with_if(self):
        import ast as _ast
        tree = _ast.parse(
            "def f(x):\n    if x > 0:\n        return 1\n    return 0"
        )
        func = tree.body[0]
        cc = _count_complexity(func)
        assert cc == 2

    def test_check_syntax_valid(self):
        ok, issues = _check_syntax("def f(): pass")
        assert ok is True

    def test_check_syntax_invalid(self):
        ok, issues = _check_syntax("def f(")
        assert ok is False

    def test_analyze_code_quality_clean(self):
        code = (
            "def clean_func(x):\n"
            '    """A clean function."""\n'
            "    return x * 2\n"
        )
        result = _analyze_code_quality(code)
        assert result["syntax_valid"] is True
        assert result["metrics"]["function_count"] == 1

    def test_analyze_code_quality_with_issues(self):
        code = (
            "def bad_func(x):\n"
            "    return x\n"
        )
        result = _analyze_code_quality(code)
        assert result["syntax_valid"] is True
        assert len(result["issues"]) > 0
