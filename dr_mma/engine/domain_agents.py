"""
Domain Agent Base — 专业 Agent 基类与领域注册表。

Phase 5: 领域专业化。每个专业 Agent 有独立的能力画像和校准任务，
与通用 Agent 可混合编排，新增不影响已有协同流程。

核心功能：
- DomainAgent 基类：统一专业 Agent 接口
- 能力画像：每个领域有独立的能力向量
- 校准任务：定期运行标准测试更新能力分数
- 领域注册表：统一管理可用专业 Agent
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DomainType(Enum):
    """专业领域类型。"""

    PAPER_WRITING = "paper_writing"
    CODE_DEVELOPMENT = "code_development"
    ENGINEERING_SIM = "engineering_simulation"
    DATA_ANALYSIS = "data_analysis"
    KNOWLEDGE_MGMT = "knowledge_management"


class CalibrationStatus(Enum):
    """校准状态。"""

    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    DEGRADED = "degraded"


@dataclass
class CapabilityProfile:
    """能力画像。"""

    domain: DomainType
    skills: dict[str, float] = field(default_factory=dict)  # skill_name -> score
    confidence: float = 0.0  # 基于样本量的可信度
    sample_count: int = 0
    failure_count: int = 0
    last_calibrated_at: Optional[float] = None
    calibration_status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED

    def to_dict(self) -> dict:
        return {
            "domain": self.domain.value,
            "skills": self.skills,
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "failure_count": self.failure_count,
            "last_calibrated_at": self.last_calibrated_at,
            "calibration_status": self.calibration_status.value,
        }

    def average_skill_score(self) -> float:
        if not self.skills:
            return 0.0
        return sum(self.skills.values()) / len(self.skills)


@dataclass
class CalibrationResult:
    """校准结果。"""

    domain: DomainType
    task_name: str
    passed: bool
    score: float
    duration_ms: float
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DomainTask:
    """领域专属任务。"""

    task_id: str
    domain_type: DomainType
    task_name: str
    objective: str
    input_data: dict = field(default_factory=dict)
    expected_output_schema: str = ""
    success_criteria: list[str] = field(default_factory=list)
    priority: float = 1.0
    metadata: dict = field(default_factory=dict)


class DomainAgent(ABC):
    """
    专业 Agent 基类。

    所有领域 Agent 必须继承此类并实现抽象方法。
    提供统一的能力画像管理、校准接口和任务执行框架。
    """

    def __init__(self, agent_id: str, domain: DomainType):
        self.agent_id = agent_id
        self.domain = domain
        self.profile = CapabilityProfile(domain=domain)
        self._calibration_history: list[CalibrationResult] = []
        self._task_history: list[dict] = []

    @abstractmethod
    def get_domain_skills(self) -> dict[str, float]:
        """返回该领域的标准技能列表及当前评分。"""
        ...

    @abstractmethod
    def get_calibration_tasks(self) -> list[dict]:
        """返回该校准任务列表（用于定期能力评估）。"""
        ...

    @abstractmethod
    def execute_domain_task(self, task: DomainTask) -> dict:
        """执行领域专属任务。"""
        ...

    @abstractmethod
    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        """验证输出是否符合领域标准。返回 (通过, 问题列表)。"""
        ...

    def calibrate(self) -> list[CalibrationResult]:
        """运行校准任务更新能力画像。"""
        tasks = self.get_calibration_tasks()
        results = []

        for task_def in tasks:
            start = time.time()
            task = DomainTask(
                task_id=f"calib_{task_def['name']}",
                domain_type=self.domain,
                task_name=task_def["name"],
                objective=task_def.get("objective", ""),
                input_data=task_def.get("input", {}),
            )

            try:
                output = self.execute_domain_task(task)
                passed, issues = self.validate_output(output)
                score = 1.0 if not issues else max(0.0, 1.0 - len(issues) * 0.1)
            except Exception as e:
                passed = False
                score = 0.0
                issues = [str(e)]

            duration = (time.time() - start) * 1000
            result = CalibrationResult(
                domain=self.domain,
                task_name=task_def["name"],
                passed=passed,
                score=score,
                duration_ms=duration,
                details={"issues": issues},
            )
            results.append(result)
            self._calibration_history.append(result)

        self._update_profile_from_results(results)
        return results

    def _update_profile_from_results(self, results: list[CalibrationResult]):
        """根据校准结果更新能力画像。"""
        if not results:
            return

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        # 更新样本量
        self.profile.sample_count += total
        self.profile.failure_count += total - passed

        # 更新技能评分（简化：按校准通过率加权）
        pass_rate = passed / total
        for skill_name in self.profile.skills:
            current = self.profile.skills[skill_name]
            # 指数移动平均
            self.profile.skills[skill_name] = current * 0.7 + pass_rate * 0.3

        # 更新置信度（样本量越大越可信）
        self.profile.confidence = min(1.0, self.profile.sample_count / 50)

        # 更新状态
        if pass_rate >= 0.8:
            self.profile.calibration_status = CalibrationStatus.CALIBRATED
        elif pass_rate >= 0.5:
            self.profile.calibration_status = CalibrationStatus.DEGRADED
        else:
            self.profile.calibration_status = CalibrationStatus.DEGRADED

        self.profile.last_calibrated_at = time.time()

    def get_profile(self) -> CapabilityProfile:
        """获取当前能力画像。"""
        return self.profile

    def calibration_summary(self) -> dict:
        """返回校准摘要。"""
        if not self._calibration_history:
            return {
                "total_runs": 0,
                "avg_score": 0.0,
                "last_run": None,
            }

        scores = [r.score for r in self._calibration_history]
        return {
            "total_runs": len(self._calibration_history),
            "avg_score": round(sum(scores) / len(scores), 4),
            "last_run": self._calibration_history[-1].timestamp,
            "recent_scores": [r.score for r in self._calibration_history[-5:]],
        }


class DomainRegistry:
    """
    领域注册表：统一管理可用专业 Agent。

    Usage:
        registry = DomainRegistry()
        registry.register(agent)

        # 按领域查询
        agents = registry.get_agents_by_domain(DomainType.CODE_DEVELOPMENT)

        # 按能力查询
        best = registry.find_best_agent("python_coding")
    """

    def __init__(self):
        self._agents: dict[str, DomainAgent] = {}
        self._domain_index: dict[DomainType, list[str]] = {d: [] for d in DomainType}

    def register(self, agent: DomainAgent) -> None:
        """注册专业 Agent。"""
        self._agents[agent.agent_id] = agent
        if agent.domain not in self._domain_index:
            self._domain_index[agent.domain] = []
        self._domain_index[agent.domain].append(agent.agent_id)

    def unregister(self, agent_id: str) -> bool:
        """注销 Agent。"""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            if agent.domain in self._domain_index:
                if agent_id in self._domain_index[agent.domain]:
                    self._domain_index[agent.domain].remove(agent_id)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[DomainAgent]:
        """获取 Agent。"""
        return self._agents.get(agent_id)

    def get_agents_by_domain(self, domain: DomainType) -> list[DomainAgent]:
        """按领域查询 Agent。"""
        ids = self._domain_index.get(domain, [])
        return [self._agents[i] for i in ids if i in self._agents]

    def find_best_agent(self, skill_name: str) -> Optional[DomainAgent]:
        """查找某项技能评分最高的 Agent。"""
        best = None
        best_score = -1.0

        for agent in self._agents.values():
            score = agent.profile.skills.get(skill_name, 0.0)
            if score > best_score:
                best_score = score
                best = agent

        return best

    def calibrate_all(self) -> dict[str, list[CalibrationResult]]:
        """校准所有注册 Agent。"""
        results = {}
        for agent in self._agents.values():
            results[agent.agent_id] = agent.calibrate()
        return results

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def registry_summary(self) -> dict:
        """返回注册表摘要。"""
        by_domain = {}
        for domain, ids in self._domain_index.items():
            agents = [self._agents[i] for i in ids if i in self._agents]
            by_domain[domain.value] = {
                "count": len(agents),
                "avg_calibration": round(
                    sum(a.profile.average_skill_score() for a in agents) / max(1, len(agents)),
                    4,
                ),
            }

        return {
            "total_agents": self.agent_count,
            "by_domain": by_domain,
        }
