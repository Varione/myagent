"""
Code Development Domain Agent.

Professional code development Agent supporting code generation, code review, and test generation subtasks.
Inherits DomainAgent base class with pure Python code quality validation (syntax, naming, complexity).
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Optional

from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainTask,
    DomainType,
)


_PYTHON_CLASS_NAME_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_PYTHON_FUNCTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PYTHON_MODULE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CONSTANT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield",
})


def _check_naming_convention(name: str, kind: str) -> list[str]:
    """Check Python identifier naming against PEP 8."""
    issues = []

    if not name:
        return ["Empty identifier"]

    if name in _PYTHON_KEYWORDS:
        issues.append(f"Name '{name}' is a Python keyword")
        return issues

    if kind == "class":
        if not _PYTHON_CLASS_NAME_RE.match(name):
            issues.append(
                f"Class name '{name}' should follow CapWords convention"
            )
    elif kind == "function":
        if not _PYTHON_FUNCTION_NAME_RE.match(name):
            issues.append(
                f"Function name '{name}' should be snake_case"
            )
    elif kind == "variable":
        if not _PYTHON_FUNCTION_NAME_RE.match(name):
            issues.append(
                f"Variable name '{name}' should be snake_case"
            )
    elif kind == "constant":
        if not _CONSTANT_NAME_RE.match(name):
            issues.append(
                f"Constant name '{name}' should be UPPER_SNAKE_CASE"
            )
    elif kind == "module":
        if not _PYTHON_MODULE_NAME_RE.match(name):
            issues.append(
                f"Module name '{name}' should be snake_case"
            )

    return issues


def _count_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a single AST node (simplified)."""
    complexity = 1

    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.While, ast.ExceptHandler),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1

    return complexity


def _count_lines(node: ast.FunctionDef) -> int:
    """Count function body lines (excluding decorators and docstrings)."""
    if not node.body:
        return 0

    start_line = node.body[0].lineno

    body_start = 0
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant,))
        and isinstance(node.body[0].value.value, str)
    ):
        body_start = 1

    if body_start >= len(node.body):
        return 0

    end_line = node.body[-1].end_lineno or node.body[-1].lineno
    return max(0, end_line - start_line)


def _check_syntax(code: str) -> tuple[bool, list[str]]:
    """Syntax check: attempt to parse Python code."""
    issues = []
    try:
        ast.parse(code)
        return True, issues
    except SyntaxError as e:
        issues.append(
            f"Syntax error at line {e.lineno}: {e.msg}"
        )
        return False, issues


def _check_docstrings(tree: ast.Module) -> list[str]:
    """Check if classes and functions contain docstrings."""
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                not node.body
                or not isinstance(node.body[0], ast.Expr)
                or not isinstance(
                    getattr(node.body[0].value, "value", None), str
                )
            ):
                issues.append(
                    f"Function '{node.name}' missing docstring"
                )
        elif isinstance(node, ast.ClassDef):
            if (
                not node.body
                or not isinstance(node.body[0], ast.Expr)
                or not isinstance(
                    getattr(node.body[0].value, "value", None), str
                )
            ):
                issues.append(
                    f"Class '{node.name}' missing docstring"
                )

    return issues


def _check_function_length(tree: ast.Module, max_lines: int = 50) -> list[str]:
    """Check if function length exceeds threshold."""
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = _count_lines(node)
            if length > max_lines:
                issues.append(
                    f"Function '{node.name}' has {length} lines "
                    f"(exceeds {max_lines})"
                )

    return issues


def _check_cyclomatic_complexity(
    tree: ast.Module, max_complexity: int = 10
) -> list[str]:
    """Check function cyclomatic complexity."""
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _count_complexity(node)
            if cc > max_complexity:
                issues.append(
                    f"Function '{node.name}' has cyclomatic complexity "
                    f"{cc} (exceeds {max_complexity})"
                )

    return issues


def _check_naming(tree: ast.Module) -> list[str]:
    """Check all identifier naming conventions."""
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            issues.extend(_check_naming_convention(node.name, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            issues.extend(_check_naming_convention(node.name, "function"))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.isupper():
                        issues.extend(
                            _check_naming_convention(target.id, "constant")
                        )
                    else:
                        issues.extend(
                            _check_naming_convention(target.id, "variable")
                        )

    return issues


def _analyze_code_quality(code: str) -> dict[str, Any]:
    """Full code quality analysis."""
    result = {
        "syntax_valid": True,
        "issues": [],
        "metrics": {},
    }

    syntax_ok, syntax_issues = _check_syntax(code)
    result["syntax_valid"] = syntax_ok
    result["issues"].extend(syntax_issues)

    if not syntax_ok:
        return result

    tree = ast.parse(code)

    doc_issues = _check_docstrings(tree)
    result["issues"].extend(doc_issues)

    length_issues = _check_function_length(tree)
    result["issues"].extend(length_issues)

    complexity_issues = _check_cyclomatic_complexity(tree)
    result["issues"].extend(complexity_issues)

    naming_issues = _check_naming(tree)
    result["issues"].extend(naming_issues)

    func_count = 0
    class_count = 0
    total_complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
            total_complexity += _count_complexity(node)
        elif isinstance(node, ast.ClassDef):
            class_count += 1

    result["metrics"] = {
        "function_count": func_count,
        "class_count": class_count,
        "avg_complexity": round(
            total_complexity / max(1, func_count), 2
        ),
        "total_complexity": total_complexity,
    }

    return result


class CodeDevAgent(DomainAgent):
    """
    Code development professional Agent.

    Supported subtask types:
    - code_generation: Generate Python code from requirements
    - code_review: Review code quality (syntax, naming, complexity)
    - test_generation: Generate unit tests for given code
    """

    def __init__(self, agent_id: str = "code_dev_1"):
        super().__init__(agent_id, DomainType.CODE_DEVELOPMENT)
        skills = self.get_domain_skills()
        self.profile.skills.update(skills)

    def get_domain_skills(self) -> dict[str, float]:
        """Return standard skill list and initial scores for code development."""
        return {
            "python_coding": 0.5,
            "code_review": 0.5,
            "test_generation": 0.5,
            "debug_analysis": 0.5,
            "architecture_design": 0.5,
        }

    def get_calibration_tasks(self) -> list[dict]:
        """Return calibration task definitions (at least 3)."""
        return [
            {
                "name": "calib_syntax_check",
                "objective": "Verify syntax validation works correctly",
                "input": {
                    "subtask_type": "code_review",
                    "code": (
                        "def valid_function(x):\n"
                        '    """Return doubled value."""\n'
                        "    return x * 2\n"
                    ),
                },
            },
            {
                "name": "calib_code_generation",
                "objective": "Verify code generation produces valid Python",
                "input": {
                    "subtask_type": "code_generation",
                    "description": "A function that adds two numbers",
                    "language": "python",
                },
            },
            {
                "name": "calib_test_generation",
                "objective": "Verify test generation produces valid test code",
                "input": {
                    "subtask_type": "test_generation",
                    "code": (
                        "def add(a, b):\n"
                        '    """Add two numbers."""\n'
                        "    return a + b\n"
                    ),
                },
            },
            {
                "name": "calib_complexity_detection",
                "objective": "Verify complexity detection identifies complex code",
                "input": {
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
            },
        ]

    def execute_domain_task(self, task: DomainTask) -> dict:
        """Execute domain-specific task."""
        subtask_type = task.input_data.get("subtask_type", "code_review")

        if subtask_type == "code_generation":
            return self._execute_code_generation(task)
        elif subtask_type == "code_review":
            return self._execute_code_review(task)
        elif subtask_type == "test_generation":
            return self._execute_test_generation(task)
        else:
            return {
                "status": "error",
                "message": f"Unknown subtask type: {subtask_type}",
                "task_id": task.task_id,
            }

    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        """Validate output meets code development domain standards."""
        issues = []

        if not isinstance(output, dict):
            return False, ["Output must be a dictionary"]

        if "status" not in output:
            issues.append("Missing 'status' field")

        if "task_id" not in output:
            issues.append("Missing 'task_id' field")

        subtask_type = output.get("subtask_type", "")

        if subtask_type == "code_generation":
            code = output.get("generated_code", "")
            if not code:
                issues.append("Generated code is empty")
            else:
                syntax_ok, syntax_issues = _check_syntax(code)
                if not syntax_ok:
                    issues.extend(syntax_issues)

        elif subtask_type == "code_review":
            analysis = output.get("analysis", {})
            if not isinstance(analysis, dict):
                issues.append("'analysis' must be a dictionary")
            else:
                if "syntax_valid" not in analysis:
                    issues.append(
                        "Analysis missing 'syntax_valid' field"
                    )
                if "issues" not in analysis:
                    issues.append("Analysis missing 'issues' field")

        elif subtask_type == "test_generation":
            test_code = output.get("generated_tests", "")
            if not test_code:
                issues.append("Generated tests are empty")
            else:
                syntax_ok, syntax_issues = _check_syntax(test_code)
                if not syntax_ok:
                    issues.extend(syntax_issues)

        passed = len(issues) == 0
        return passed, issues

    def _execute_code_generation(self, task: DomainTask) -> dict:
        """Code generation subtask."""
        description = task.input_data.get("description", "")

        generated_code = self._generate_code_from_description(description)
        analysis = _analyze_code_quality(generated_code)

        return {
            "status": "completed",
            "subtask_type": "code_generation",
            "task_id": task.task_id,
            "description": description,
            "generated_code": generated_code,
            "self_analysis": analysis,
            "timestamp": time.time(),
        }

    def _execute_code_review(self, task: DomainTask) -> dict:
        """Code review subtask."""
        code = task.input_data.get("code", "")

        if not code:
            return {
                "status": "error",
                "subtask_type": "code_review",
                "task_id": task.task_id,
                "message": "No code provided for review",
                "analysis": {"syntax_valid": False, "issues": [], "metrics": {}},
            }

        analysis = _analyze_code_quality(code)

        issue_count = len(analysis["issues"])
        if not analysis["syntax_valid"]:
            quality_score = 0.0
        else:
            quality_score = max(0.0, 1.0 - issue_count * 0.1)

        return {
            "status": "completed",
            "subtask_type": "code_review",
            "task_id": task.task_id,
            "analysis": analysis,
            "quality_score": round(quality_score, 4),
            "recommendations": self._generate_recommendations(
                analysis["issues"]
            ),
            "timestamp": time.time(),
        }

    def _execute_test_generation(self, task: DomainTask) -> dict:
        """Test generation subtask."""
        code = task.input_data.get("code", "")

        if not code:
            return {
                "status": "error",
                "subtask_type": "test_generation",
                "task_id": task.task_id,
                "message": "No code provided for test generation",
                "generated_tests": "",
            }

        test_code = self._generate_tests_from_code(code)
        analysis = _analyze_code_quality(test_code)

        return {
            "status": "completed",
            "subtask_type": "test_generation",
            "task_id": task.task_id,
            "source_code": code,
            "generated_tests": test_code,
            "self_analysis": analysis,
            "timestamp": time.time(),
        }

    def _generate_code_from_description(self, description: str) -> str:
        """Generate Python code from natural language description."""
        desc_lower = description.lower()

        if "add" in desc_lower or "sum" in desc_lower:
            return (
                'def add(a, b):\n'
                '    """Add two numbers and return the result."""\n'
                "    return a + b\n"
            )

        if "fibonacci" in desc_lower:
            return (
                'def fibonacci(n):\n'
                '    """Return the nth Fibonacci number."""\n'
                "    if n <= 0:\n"
                "        return 0\n"
                "    if n == 1:\n"
                "        return 1\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
            )

        if "sort" in desc_lower:
            return (
                'def bubble_sort(arr):\n'
                '    """Sort a list using bubble sort algorithm."""\n'
                "    n = len(arr)\n"
                "    for i in range(n):\n"
                "        for j in range(0, n - i - 1):\n"
                "            if arr[j] > arr[j + 1]:\n"
                "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
                "    return arr\n"
            )

        func_name = description.replace(" ", "_").lower()[:30] or "my_function"
        if not func_name[0].isalpha():
            func_name = "func_" + func_name

        return (
            f'def {func_name}(*args, **kwargs):\n'
            '    """TODO: Implement this function."""\n'
            "    pass\n"
        )

    def _generate_tests_from_code(self, code: str) -> str:
        """Generate unit tests from given code."""
        if not code:
            return ""

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "# Error: Source code has syntax errors\n"

        test_lines = [
            "import unittest",
            "",
            "",
            "class GeneratedTests(unittest.TestCase):",
            '    """Auto-generated unit tests."""',
            "",
        ]

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                test_lines.extend(
                    self._create_test_method(node)
                )

        test_lines.append("")
        test_lines.append("")
        test_lines.append('if __name__ == "__main__":')
        test_lines.append("    unittest.main()")

        return "\n".join(test_lines)

    def _create_test_method(
        self, func_node: ast.FunctionDef
    ) -> list[str]:
        """Generate test method for a single function."""
        args = []
        for arg in func_node.args.args:
            if arg.arg == "self":
                continue
            args.append(arg.arg)

        test_name = f"test_{func_node.name}"
        lines = [
            f"    def {test_name}(self):",
            f'        """Test {func_node.name} function."""',
        ]

        if not args:
            lines.append(f"        result = {func_node.name}()")
            lines.append("        self.assertIsNotNone(result)")
        else:
            call_args = ", ".join(
                f"{a}=0" if a != "n" else f"{a}=1" for a in args
            )
            lines.append(f"        result = {func_node.name}({call_args})")
            lines.append("        self.assertIsNotNone(result)")

        lines.append("")
        lines.append("")
        return lines

    def _generate_recommendations(self, issues: list[str]) -> list[str]:
        """Generate improvement recommendations from issue list."""
        recommendations = []

        for issue in issues:
            if "missing docstring" in issue.lower():
                recommendations.append(
                    "Add docstrings to all public functions and classes"
                )
            elif "cyclomatic complexity" in issue.lower():
                recommendations.append(
                    "Refactor complex functions into smaller, focused units"
                )
            elif "lines" in issue.lower() and "exceeds" in issue.lower():
                recommendations.append(
                    "Split long functions into smaller helper functions"
                )
            elif "should follow" in issue.lower() or "should be" in issue.lower():
                recommendations.append(
                    "Follow PEP 8 naming conventions consistently"
                )
            elif "syntax error" in issue.lower():
                recommendations.append(
                    "Fix syntax errors before further analysis"
                )

        return list(dict.fromkeys(recommendations))
