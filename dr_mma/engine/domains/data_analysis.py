"""
Data Analysis Domain Agent.

Professional data analysis Agent supporting descriptive statistics, hypothesis testing,
and report generation subtasks. Uses only Python standard library (math module).
Inherits DomainAgent base class with pure Python statistical computations.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainTask,
    DomainType,
)


def _compute_mean(data: list[float]) -> float:
    """Compute arithmetic mean of a list of numbers."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def _compute_median(data: list[float]) -> float:
    """Compute median of a list of numbers without sorting in place."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_data[mid])
    else:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


def _compute_std_dev(data: list[float], population: bool = False) -> float:
    """Compute standard deviation. Use population=True for population std dev."""
    n = len(data)
    if n < 2:
        return 0.0
    mean = _compute_mean(data)
    divisor = n if population else (n - 1)
    variance = sum((x - mean) ** 2 for x in data) / divisor
    return math.sqrt(variance)


def _compute_quartile(sorted_data: list[float], q: float) -> float:
    """
    Compute the q-th quartile (q in 0..1) using linear interpolation.
    
    Uses the method where Q1 = 25th percentile, Q3 = 75th percentile.
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_data[0])
    
    # Position in the sorted array (0-indexed, fractional)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    
    if lo == hi:
        return float(sorted_data[lo])
    
    frac = pos - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _compute_descriptive_stats(data: list[float]) -> dict[str, Any]:
    """Compute full descriptive statistics for a dataset."""
    if not data:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": None,
            "max": None,
            "q1": 0.0,
            "q3": 0.0,
            "count": 0,
        }
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    return {
        "mean": round(_compute_mean(data), 6),
        "median": round(_compute_median(data), 6),
        "std_dev": round(_compute_std_dev(data), 6),
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "q1": round(_compute_quartile(sorted_data, 0.25), 6),
        "q3": round(_compute_quartile(sorted_data, 0.75), 6),
        "count": n,
    }


def _t_test_independent(group_a: list[float], group_b: list[float]) -> dict[str, Any]:
    """
    Perform an independent two-sample t-test (Welch's approximation).
    
    Returns t-statistic, degrees of freedom, and significance assessment.
    Uses a simplified approach: compares |t| against common critical values.
    """
    n_a = len(group_a)
    n_b = len(group_b)
    
    if n_a < 2 or n_b < 2:
        return {
            "t_statistic": 0.0,
            "df": 0,
            "significant": False,
            "conclusion": "Insufficient data for t-test (need at least 2 samples per group)",
        }
    
    mean_a = _compute_mean(group_a)
    mean_b = _compute_mean(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)
    
    se_a = var_a / n_a
    se_b = var_b / n_b
    se_diff = math.sqrt(se_a + se_b)
    
    if se_diff == 0:
        return {
            "t_statistic": 0.0,
            "df": min(n_a - 1, n_b - 1),
            "significant": False,
            "conclusion": "Zero variance in both groups; t-statistic undefined",
        }
    
    t_stat = (mean_a - mean_b) / se_diff
    
    # Welch-Satterthwaite degrees of freedom
    numerator = (se_a + se_b) ** 2
    denominator = (se_a ** 2 / (n_a - 1)) + (se_b ** 2 / (n_b - 1))
    
    if denominator == 0:
        df = min(n_a - 1, n_b - 1)
    else:
        df = numerator / denominator
    
    # Approximate significance using critical t-values for alpha=0.05
    # For large df, critical value is ~1.96 (z-score). For small df, higher.
    abs_t = abs(t_stat)
    
    # Critical values for two-tailed test at alpha=0.05
    # Approximation: t_crit decreases as df increases
    if df < 5:
        t_critical = 4.604
    elif df < 10:
        t_critical = 2.262
    elif df < 30:
        t_critical = 2.048
    else:
        t_critical = 1.96
    
    significant = abs_t > t_critical
    
    if significant:
        conclusion = (
            f"Reject null hypothesis: significant difference detected "
            f"(t={t_stat:.4f}, df={df:.1f}, |t|>{t_critical})"
        )
    else:
        conclusion = (
            f"Fail to reject null hypothesis: no significant difference "
            f"(t={t_stat:.4f}, df={df:.1f}, |t|<={t_critical})"
        )
    
    return {
        "t_statistic": round(t_stat, 6),
        "df": round(df, 2),
        "significant": significant,
        "conclusion": conclusion,
    }


def _generate_report(stats: dict[str, Any]) -> dict[str, Any]:
    """Generate a text report from statistics results."""
    sections = []
    
    # Section 1: Overview
    overview_parts = []
    count = stats.get("count", 0)
    if count > 0:
        overview_parts.append(f"Dataset contains {count} observations.")
    else:
        overview_parts.append("Dataset is empty or no data provided.")
    
    mean_val = stats.get("mean")
    median_val = stats.get("median")
    if mean_val is not None and median_val is not None:
        overview_parts.append(
            f"Central tendency: mean={mean_val}, median={median_val}."
        )
    
    sections.append("Overview")
    report_lines = ["=== Data Analysis Report ===", "", "1. Overview", ""]
    report_lines.extend(overview_parts)
    report_lines.append("")
    
    # Section 2: Distribution
    std_dev = stats.get("std_dev")
    min_val = stats.get("min")
    max_val = stats.get("max")
    q1 = stats.get("q1")
    q3 = stats.get("q3")
    
    dist_parts = []
    if std_dev is not None:
        dist_parts.append(f"Standard deviation: {std_dev}")
    if min_val is not None and max_val is not None:
        dist_parts.append(f"Range: [{min_val}, {max_val}]")
        iqr = max_val - min_val if isinstance(max_val, (int, float)) and isinstance(min_val, (int, float)) else None
        if iqr is not None:
            dist_parts.append(f"Total range span: {iqr}")
    if q1 is not None and q3 is not None:
        inter_qr = q3 - q1
        dist_parts.append(f"Interquartile range (Q1={q1}, Q3={q3}): {inter_qr}")
    
    sections.append("Distribution")
    report_lines.append("2. Distribution")
    report_lines.append("")
    report_lines.extend(dist_parts)
    report_lines.append("")
    
    # Section 3: Interpretation
    interp_parts = []
    if mean_val is not None and median_val is not None:
        skew_hint = mean_val - median_val
        if abs(skew_hint) < 1e-9:
            interp_parts.append("Mean equals median, suggesting symmetric distribution.")
        elif skew_hint > 0:
            interp_parts.append(
                f"Mean ({mean_val}) exceeds median ({median_val}), "
                f"suggesting right-skewed distribution."
            )
        else:
            interp_parts.append(
                f"Median ({median_val}) exceeds mean ({mean_val}), "
                f"suggesting left-skewed distribution."
            )
    
    if std_dev is not None and mean_val is not None and mean_val != 0:
        cv = abs(std_dev / mean_val) * 100
        interp_parts.append(f"Coefficient of variation: {cv:.2f}%")
        if cv > 100:
            interp_parts.append("High variability relative to the mean.")
        elif cv < 30:
            interp_parts.append("Low variability; data is tightly clustered around the mean.")
    
    sections.append("Interpretation")
    report_lines.append("3. Interpretation")
    report_lines.append("")
    report_lines.extend(interp_parts)
    report_lines.append("")
    report_lines.append("=== End of Report ===")
    
    return {
        "report_text": "\n".join(report_lines),
        "sections": sections,
    }


class DataAnalysisAgent(DomainAgent):
    """
    Data analysis professional Agent.

    Supported subtask types:
    - descriptive_stats: Compute mean, median, std_dev, min, max, q1, q3
    - hypothesis_testing: Independent two-sample t-test
    - report_generation: Generate text report from statistics
    """

    def __init__(self, agent_id: str = "data_analysis_1"):
        super().__init__(agent_id, DomainType.DATA_ANALYSIS)
        skills = self.get_domain_skills()
        self.profile.skills.update(skills)

    def get_domain_skills(self) -> dict[str, float]:
        """Return standard skill list and initial scores for data analysis."""
        return {
            "statistical_modeling": 0.8,
            "data_visualization": 0.75,
            "report_generation": 0.85,
            "hypothesis_testing": 0.7,
            "descriptive_stats": 0.9,
        }

    def get_calibration_tasks(self) -> list[dict]:
        """Return calibration task definitions covering all subtask types."""
        return [
            {
                "name": "calib_descriptive_stats",
                "objective": "Verify descriptive statistics computation is correct",
                "input": {
                    "subtask": "descriptive_stats",
                    "data": [2, 4, 6, 8, 10],
                },
            },
            {
                "name": "calib_hypothesis_testing",
                "objective": "Verify t-test implementation produces valid results",
                "input": {
                    "subtask": "hypothesis_testing",
                    "group_a": [10, 12, 14, 16, 18],
                    "group_b": [5, 7, 9, 11, 13],
                },
            },
            {
                "name": "calib_report_generation",
                "objective": "Verify report generation produces structured output",
                "input": {
                    "subtask": "report_generation",
                    "stats": {
                        "mean": 6.0,
                        "median": 6.0,
                        "std_dev": 2.828427,
                        "min": 2,
                        "max": 10,
                        "q1": 4.0,
                        "q3": 8.0,
                        "count": 5,
                    },
                },
            },
            {
                "name": "calib_single_element_stats",
                "objective": "Verify edge case handling for single-element data",
                "input": {
                    "subtask": "descriptive_stats",
                    "data": [42],
                },
            },
        ]

    def execute_domain_task(self, task: DomainTask) -> dict:
        """Execute domain-specific task by routing to subtask handler."""
        subtask = task.input_data.get("subtask", "")

        if subtask == "descriptive_stats":
            return self._execute_descriptive_stats(task)
        elif subtask == "hypothesis_testing":
            return self._execute_hypothesis_testing(task)
        elif subtask == "report_generation":
            return self._execute_report_generation(task)
        else:
            return {
                "status": "error",
                "message": f"Unknown subtask type: {subtask}",
                "task_id": task.task_id,
            }

    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        """Validate output meets data analysis domain standards."""
        issues = []

        if not isinstance(output, dict):
            return False, ["Output must be a dictionary"]

        # Check common fields
        if "task_id" not in output:
            issues.append("Missing task_id field")

        # Determine subtask type from input_data if available, else from output
        subtask = output.get("subtask", "")

        if subtask == "descriptive_stats":
            required_keys = [
                "mean", "median", "std_dev", "min", "max", "q1", "q3", "count"
            ]
            for key in required_keys:
                if key not in output:
                    issues.append(f"Missing required key: {key}")

        elif subtask == "hypothesis_testing":
            required_keys = ["t_statistic", "df", "significant", "conclusion"]
            for key in required_keys:
                if key not in output:
                    issues.append(f"Missing required key: {key}")
            if "significant" in output and not isinstance(output["significant"], bool):
                issues.append("significant must be a boolean")

        elif subtask == "report_generation":
            required_keys = ["report_text", "sections"]
            for key in required_keys:
                if key not in output:
                    issues.append(f"Missing required key: {key}")
            if "report_text" in output and not isinstance(output["report_text"], str):
                issues.append("report_text must be a string")
            if "sections" in output and not isinstance(output["sections"], list):
                issues.append("sections must be a list")

        passed = len(issues) == 0
        return passed, issues

    def _execute_descriptive_stats(self, task: DomainTask) -> dict:
        """Compute descriptive statistics for the provided data."""
        data = task.input_data.get("data", [])

        if not isinstance(data, list):
            return {
                "status": "error",
                "subtask": "descriptive_stats",
                "task_id": task.task_id,
                "message": "Input data must be a list of numbers",
            }

        # Validate all elements are numeric
        for i, val in enumerate(data):
            if not isinstance(val, (int, float)):
                return {
                    "status": "error",
                    "subtask": "descriptive_stats",
                    "task_id": task.task_id,
                    "message": f"Element at index {i} is not numeric: {val}",
                }

        stats = _compute_descriptive_stats(data)

        return {
            "status": "completed",
            "subtask": "descriptive_stats",
            "task_id": task.task_id,
            **stats,
            "timestamp": time.time(),
        }

    def _execute_hypothesis_testing(self, task: DomainTask) -> dict:
        """Perform independent two-sample t-test."""
        group_a = task.input_data.get("group_a", [])
        group_b = task.input_data.get("group_b", [])

        if not group_a or not group_b:
            return {
                "status": "error",
                "subtask": "hypothesis_testing",
                "task_id": task.task_id,
                "message": "Both group_a and group_b must be non-empty lists",
                "t_statistic": 0.0,
                "df": 0,
                "significant": False,
                "conclusion": "Cannot perform t-test with empty groups",
            }

        result = _t_test_independent(group_a, group_b)

        return {
            "status": "completed",
            "subtask": "hypothesis_testing",
            "task_id": task.task_id,
            **result,
            "timestamp": time.time(),
        }

    def _execute_report_generation(self, task: DomainTask) -> dict:
        """Generate a text report from statistics."""
        stats = task.input_data.get("stats", {})

        if not isinstance(stats, dict):
            return {
                "status": "error",
                "subtask": "report_generation",
                "task_id": task.task_id,
                "message": "Input stats must be a dictionary",
                "report_text": "",
                "sections": [],
            }

        report = _generate_report(stats)

        return {
            "status": "completed",
            "subtask": "report_generation",
            "task_id": task.task_id,
            **report,
            "timestamp": time.time(),
        }


