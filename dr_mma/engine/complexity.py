"""Task complexity evaluation and mode routing."""

from dataclasses import dataclass
import re


MODE_DIRECT = "Direct Mode"
MODE_SINGLE_REVIEW = "Single Review Mode"
MODE_COMPACT = "Compact Mode"
MODE_STANDARD = "Standard Mode"
MODE_EXPANDED = "Expanded Mode"


@dataclass
class ComplexityReport:
    score: int
    mode: str
    step_count: int
    domain_depth: int
    tool_requirement: int
    verification_need: int
    output_risk: int
    context_length: int
    rationale: str


class TaskComplexityEvaluator:
    """Heuristic evaluator used to choose the collaboration mode."""

    def evaluate(self, task_text: str) -> ComplexityReport:
        text = task_text.strip()
        explicit_direct = any(token in text for token in ["简单", "一句话", "直接回答", "简答"])
        step_count = self._score_step_count(text)
        domain_depth = self._score_domain_depth(text)
        tool_requirement = self._score_tool_requirement(text)
        verification_need = self._score_verification_need(text)
        output_risk = self._score_output_risk(text)
        context_length = self._score_context_length(text)

        score = (
            step_count
            + domain_depth
            + tool_requirement
            + verification_need
            + output_risk
            + context_length
        )
        mode = self._map_mode(score, explicit_direct=explicit_direct)
        rationale = (
            f"steps={step_count}, domain={domain_depth}, tools={tool_requirement}, "
            f"verify={verification_need}, risk={output_risk}, context={context_length}"
        )
        return ComplexityReport(
            score=score,
            mode=mode,
            step_count=step_count,
            domain_depth=domain_depth,
            tool_requirement=tool_requirement,
            verification_need=verification_need,
            output_risk=output_risk,
            context_length=context_length,
            rationale=rationale,
        )

    def _score_step_count(self, text: str) -> int:
        score = 1
        markers = ["并", "同时", "先", "然后", "最后", "步骤", "阶段", "拆解", "设计"]
        hits = sum(1 for marker in markers if marker in text)
        if hits >= 4:
            score = 3
        elif hits >= 2:
            score = 2
        return score

    def _score_domain_depth(self, text: str) -> int:
        keywords = [
            "架构", "算法", "数据库", "权限", "同步", "分布式", "协议", "实时", "安全", "模型",
        ]
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits >= 5:
            return 3
        if hits >= 2:
            return 2
        return 1 if hits else 0

    def _score_tool_requirement(self, text: str) -> int:
        keywords = ["代码", "文件", "数据库", "接口", "API", "检索", "测试", "运行"]
        hits = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        if hits >= 4:
            return 3
        if hits >= 2:
            return 2
        return 1 if hits else 0

    def _score_verification_need(self, text: str) -> int:
        keywords = ["验证", "校验", "审查", "review", "测试", "正确", "风险"]
        hits = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        if hits >= 3:
            return 2
        return 1 if hits else 0

    def _score_output_risk(self, text: str) -> int:
        keywords = ["生产", "线上", "权限", "安全", "财务", "医疗", "合规"]
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits >= 2:
            return 2
        return 1 if hits else 0

    def _score_context_length(self, text: str) -> int:
        lines = len([line for line in re.split(r"\r?\n", text) if line.strip()])
        if len(text) > 600 or lines > 12:
            return 2
        return 1 if len(text) > 200 or lines > 5 else 0

    def _map_mode(self, score: int, explicit_direct: bool = False) -> str:
        if explicit_direct and score <= 2:
            return MODE_DIRECT
        if score <= 5:
            return MODE_SINGLE_REVIEW
        if score <= 8:
            return MODE_COMPACT
        if score <= 12:
            return MODE_STANDARD
        return MODE_EXPANDED
