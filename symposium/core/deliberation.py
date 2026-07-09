"""
圆桌研讨引擎 - 多模型多轮对话式研讨

工作方式：
  1. 每个模型看到此前所有模型的发言历史
  2. 在之前的分析基础上补充、反驳、深化
  3. 经过 N 轮，各模型对任务形成共识/分歧
  4. 全部记录写入消息总线（debate 频道）
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from ..models.base import BaseModel, ChatMessage
from .message_bus import MessageBus, CHANNEL_DEBATE


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────


@dataclass
class ModelContribution:
    """单个模型在某轮研讨中的贡献"""
    model_name: str
    round_number: int
    content: str
    elapsed: float = 0.0
    tokens_used: int = 0

    def summary(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        if len(self.content) > 60:
            preview += "…"
        return f"[Round {self.round_number}] {self.model_name}: {preview}"


@dataclass
class DeliberationResult:
    """整场研讨的结果"""
    task: str
    num_rounds: int
    contributions: list[ModelContribution] = field(default_factory=list)
    total_time: float = 0.0

    def get_contributions(self, model_name: Optional[str] = None) -> list[ModelContribution]:
        if model_name:
            return [c for c in self.contributions if c.model_name == model_name]
        return list(self.contributions)

    def get_round(self, n: int) -> list[ModelContribution]:
        return [c for c in self.contributions if c.round_number == n]

    def summary(self) -> str:
        lines = [
            f"圆桌研讨完成: {len(self.contributions)} 次发言, {self.num_rounds} 轮",
            f"参与模型: {list(dict.fromkeys(c.model_name for c in self.contributions))}",
            f"总耗时: {self.total_time:.1f}s",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 研讨引擎
# ──────────────────────────────────────────────


class DeliberationEngine:
    """圆桌研讨引擎"""

    def __init__(
        self,
        models: list[BaseModel],
        task: str,
        num_rounds: int = 3,
        message_bus: Optional[MessageBus] = None,
        task_context: Optional[str] = None,
    ):
        if len(models) < 2:
            raise ValueError("研讨至少需要 2 个模型")

        self.models = models
        self.task = task
        self.num_rounds = num_rounds
        self.bus = message_bus or MessageBus()
        self.task_context = task_context or ""

        # 为每个模型生成角色身份
        self._personas = self._generate_personas()

    def _generate_personas(self) -> dict[str, str]:
        """根据模型数量自动分配研讨角色"""
        role_descriptions = [
            "你是架构师，擅长从全局视角分析问题，关注系统结构、可扩展性和长期影响。"
            "你要提出宏观方案，并在其他人发言后补充架构层面的思考。",

            "你是实现者，关注具体落地细节和技术可行性。"
            "你会指出方案中的实现难点、依赖关系和工程代价。",

            "你是审查者，习惯批判性思考。"
            "你会找出方案中的漏洞、边界情况和潜在风险，提出改进建议。",

            "你是创新者，思维活跃，喜欢提出非常规的替代方案。"
            "你会挑战既有假设，引入跨领域思路。",

            "你是数据驱动者，关注量化指标和可验证性。"
            "你会追问如何度量成功、需要哪些数据支撑决策。",
        ]
        personas = {}
        for i, model in enumerate(self.models):
            desc = role_descriptions[i % len(role_descriptions)]
            personas[model.name] = desc
        return personas

    # ── 运行研讨 ──

    def run(self) -> DeliberationResult:
        """执行完整研讨流程"""
        t0 = time.time()
        result = DeliberationResult(task=self.task, num_rounds=self.num_rounds)

        # 广播研讨开始信号
        opening = (
            f"【研讨开始】任务: {self.task}\n\n"
            f"参与模型: {', '.join(m.name for m in self.models)}\n"
            f"研讨轮数: {self.num_rounds}\n"
        )
        if self.task_context:
            opening += f"\n背景信息:\n{self.task_context}"
        self.bus.publish(CHANNEL_DEBATE, "★ 系统", opening)

        for round_num in range(1, self.num_rounds + 1):
            for model in self.models:
                contribution = self._run_one_turn(model, round_num)
                result.contributions.append(contribution)

        result.total_time = time.time() - t0

        # 广播研讨结束
        closing = (
            f"【研讨结束】共 {len(result.contributions)} 次发言, "
            f"{self.num_rounds} 轮, 耗时 {result.total_time:.1f}s\n"
            "主模型即将进行汇总决策…"
        )
        self.bus.publish(CHANNEL_DEBATE, "★ 系统", closing)

        return result

    def _run_one_turn(self, model: BaseModel, round_num: int) -> ModelContribution:
        """让一个模型发表一轮见解"""
        # 1. 构建上下文：系统角色 + 任务 + 此前全部发言
        messages = [
            ChatMessage(role="system", content=self._personas.get(model.name, "")),
            ChatMessage(role="user", content=f"## 研讨任务\n\n{self.task}"),
        ]

        # 加入背景信息
        if self.task_context:
            messages.append(
                ChatMessage(role="user", content=f"## 背景信息\n\n{self.task_context}")
            )

        # 加入当前轮次的定位说明
        round_instruction = (
            f"## 第 {round_num}/{self.num_rounds} 轮研讨\n\n"
        )

        if round_num == 1:
            round_instruction += (
                "这是第一轮。请发表你对任务的分析和初步方案。\n"
                "关注点：问题理解、关键挑战、你的核心建议。"
            )
        else:
            round_instruction += (
                "请参考前面所有模型的发言，在此基础上：\n"
                "1. 指出你同意的观点并补充理由\n"
                "2. 指出你不同意的观点并说明原因\n"
                "3. 提出新的见解或深化已有分析"
            )
        messages.append(ChatMessage(role="user", content=round_instruction))

        # 2. 加入历史发言（此前所有模型 + 当前模型之前的发言）
        history = self.bus.get_channel_history(CHANNEL_DEBATE)
        for msg in history:
            if msg.sender == "★ 系统":
                messages.append(ChatMessage(role="user", content=msg.content))
            else:
                speaker = msg.sender
                messages.append(
                    ChatMessage(role="user", content=f"[{speaker}]:\n{msg.content}")
                )

        # 3. 获取模型响应
        t0 = time.time()
        response = model.chat(messages)
        elapsed = time.time() - t0

        # 4. 发布到消息总线
        self.bus.publish(
            CHANNEL_DEBATE,
            model.name,
            response.content,
            metadata={"round": round_num, "elapsed": round(elapsed, 2)},
        )

        return ModelContribution(
            model_name=model.name,
            round_number=round_num,
            content=response.content,
            elapsed=elapsed,
            tokens_used=response.usage.get("total_tokens", 0),
        )
