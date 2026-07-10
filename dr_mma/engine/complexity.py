"""Task complexity evaluation and mode routing.

六维评分模型（架构计划定义）：

TaskComplexity =
  a1 * StepCount        (1~3)
  + a2 * DomainDepth    (0~3)
  + a3 * ToolRequirement (0~3)
  + a4 * VerificationNeed (0~2)
  + a5 * OutputRisk     (0~2)
  + a6 * ContextLength  (0~2)

模式映射：
  0~2   → Direct Mode
  3~5   → Single Review Mode
  6~8   → Compact Mode
  9~12  → Standard Mode
  12+   → Expanded Mode
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


MODE_DIRECT = "Direct Mode"
MODE_SINGLE_REVIEW = "Single Review Mode"
MODE_COMPACT = "Compact Mode"
MODE_STANDARD = "Standard Mode"
MODE_EXPANDED = "Expanded Mode"

# 默认权重系数
DEFAULT_WEIGHTS = {
    "step_count": 1.0,
    "domain_depth": 1.2,
    "tool_requirement": 1.3,
    "verification_need": 1.0,
    "output_risk": 1.5,
    "context_length": 0.8,
}


@dataclass
class ComplexityReport:
    score: float = 0.0
    raw_score: int = 0
    mode: str = ""
    step_count: int = 0
    domain_depth: int = 0
    tool_requirement: int = 0
    verification_need: int = 0
    output_risk: int = 0
    context_length: int = 0
    rationale: str = ""
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "raw_score": self.raw_score,
            "mode": self.mode,
            "step_count": self.step_count,
            "domain_depth": self.domain_depth,
            "tool_requirement": self.tool_requirement,
            "verification_need": self.verification_need,
            "output_risk": self.output_risk,
            "context_length": self.context_length,
            "rationale": self.rationale,
            "weights_used": self.weights_used,
        }


class TaskComplexityEvaluator:
    """六维评分复杂度评估器，支持可配置权重。"""

    def __init__(self, weights: Optional[dict] = None):
        self._weights = weights or dict(DEFAULT_WEIGHTS)

    @property
    def weights(self) -> dict:
        return dict(self._weights)

    def set_weight(self, dimension: str, value: float):
        """设置某维度的权重。"""
        if dimension in self._weights:
            self._weights[dimension] = value

    def evaluate(self, task_text: str) -> ComplexityReport:
        text = task_text.strip()
        explicit_direct = any(
            token in text for token in ["简单", "一句话", "直接回答", "简答"]
        )

        step_count = self._score_step_count(text)
        domain_depth = self._score_domain_depth(text)
        tool_requirement = self._score_tool_requirement(text)
        verification_need = self._score_verification_need(text)
        output_risk = self._score_output_risk(text)
        context_length = self._score_context_length(text)

        raw_score = (
            step_count
            + domain_depth
            + tool_requirement
            + verification_need
            + output_risk
            + context_length
        )

        # 加权评分
        weighted_score = (
            self._weights["step_count"] * step_count
            + self._weights["domain_depth"] * domain_depth
            + self._weights["tool_requirement"] * tool_requirement
            + self._weights["verification_need"] * verification_need
            + self._weights["output_risk"] * output_risk
            + self._weights["context_length"] * context_length
        )

        mode = self._map_mode(weighted_score, explicit_direct=explicit_direct)
        rationale = (
            f"steps={step_count}(w{self._weights['step_count']}), "
            f"domain={domain_depth}(w{self._weights['domain_depth']}), "
            f"tools={tool_requirement}(w{self._weights['tool_requirement']}), "
            f"verify={verification_need}(w{self._weights['verification_need']}), "
            f"risk={output_risk}(w{self._weights['output_risk']}), "
            f"context={context_length}(w{self._weights['context_length']})"
        )

        return ComplexityReport(
            score=weighted_score,
            raw_score=raw_score,
            mode=mode,
            step_count=step_count,
            domain_depth=domain_depth,
            tool_requirement=tool_requirement,
            verification_need=verification_need,
            output_risk=output_risk,
            context_length=context_length,
            rationale=rationale,
            weights_used=dict(self._weights),
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

    def _map_mode(self, score: float, explicit_direct: bool = False) -> str:
        if explicit_direct and score <= 2:
            return MODE_DIRECT
        if score <= 5:
            return MODE_SINGLE_REVIEW
        if score <= 8:
            return MODE_COMPACT
        if score <= 12:
            return MODE_STANDARD
        return MODE_EXPANDED
