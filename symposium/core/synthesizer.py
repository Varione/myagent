"""
主模型汇总决策 - 分析研讨记录，识别共识与分歧，产出可执行计划

职责：
  1. 读取全部研讨记录
  2. 识别共识点和分歧点
  3. 裁决分歧
  4. 输出结构化的执行计划（任务分解 + 依赖关系 + 分配）
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..models.base import BaseModel, ChatMessage
from .deliberation import DeliberationResult
from .message_bus import MessageBus, CHANNEL_DEBATE, CHANNEL_CONTROL


# ──────────────────────────────────────────────
# 执行计划数据结构
# ──────────────────────────────────────────────


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class Task:
    """执行计划中的单个任务"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = ""
    description: str = ""
    assigned_model: str = ""
    dependencies: list[str] = field(default_factory=list)  # task id 列表
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Task({self.title}, {self.status.value}, →{self.assigned_model})"


@dataclass
class ExecutionPlan:
    """完整的执行计划"""
    task_title: str = ""
    tasks: list[Task] = field(default_factory=list)
    consensus_points: list[str] = field(default_factory=list)
    disagreement_points: list[dict] = field(default_factory=list)
    rationale: str = ""  # 主模型的决策理由

    def add_task(self, task: Task):
        self.tasks.append(task)

    def get_ready_tasks(self) -> list[Task]:
        """获取所有依赖已满足的待执行任务"""
        done_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in done_ids for dep in t.dependencies)
        ]

    def all_done(self) -> bool:
        return all(t.status == TaskStatus.DONE for t in self.tasks)

    def summary(self) -> str:
        return (
            f"执行计划: {len(self.tasks)} 个任务\n"
            f"共识点: {len(self.consensus_points)} 个\n"
            f"分歧点: {len(self.disagreement_points)} 个"
        )


# ──────────────────────────────────────────────
# 主模型汇总器
# ──────────────────────────────────────────────


MAIN_MODEL_SYSTEM_PROMPT = """你是 Symposium 架构中的"主模型"（Main Model），职责是从圆桌研讨中提炼共识、裁决分歧、输出可执行的协同任务计划。

你的工作流程：
1. 阅读全部研讨记录（多个模型的多轮对话）
2. 分析并列出所有共识点（模型们一致认同的观点）
3. 分析并列出所有分歧点（模型们有冲突的观点），附上各方案由
4. 对每个分歧点做出裁决，给出你的决策理由
5. 输出完整的任务分解计划，每个任务包含：标题、描述、指派给哪个模型、依赖哪些前置任务

输出格式要求：
先输出分析部分（共识 + 分歧），然后输出 JSON 格式的任务列表。
JSON 部分必须能被 json.loads() 正确解析。
格式如下：

## 共识分析
- 共识1...
- 共识2...

## 分歧与裁决
- 分歧1: ... → 裁决: ... 理由: ...

## 执行计划
```json
[
  {
    "title": "xxx",
    "description": "xxx",
    "assigned_model": "模型名称",
    "dependencies": []
  },
  ...
]
```"""


class MainModelSynthesizer:
    """主模型汇总器"""

    def __init__(self, main_model: BaseModel, message_bus: MessageBus):
        self.main_model = main_model
        self.bus = message_bus

    def synthesize(self, deliberation_result: DeliberationResult) -> ExecutionPlan:
        """分析研讨记录并生成执行计划"""
        # 1. 收集研讨全文
        debate_transcript = self.bus.format_full_debate()
        model_names = list(
            dict.fromkeys(c.model_name for c in deliberation_result.contributions)
        )

        # 2. 构建提示
        user_prompt = (
            f"## 原始任务\n\n{deliberation_result.task}\n\n"
            f"## 参与研讨的模型\n\n{', '.join(model_names)}\n\n"
            f"## 研讨完整记录\n\n{debate_transcript}\n\n"
            "请按你的系统提示要求，分析共识与分歧，输出执行计划。"
        )

        messages = [
            ChatMessage(role="system", content=MAIN_MODEL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        # 3. 调用主模型
        t0 = time.time()
        response = self.main_model.chat(messages)
        elapsed = time.time() - t0

        # 4. 发布到 control 频道
        self.bus.publish(
            CHANNEL_CONTROL,
            f"★ 主模型({self.main_model.name})",
            response.content,
            metadata={"elapsed": round(elapsed, 2)},
        )

        # 5. 解析结果
        plan = self._parse_plan(response.content)
        plan.task_title = deliberation_result.task

        # 6. 记录统计
        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"主模型汇总完成，产出 {len(plan.tasks)} 个任务，"
            f"{len(plan.consensus_points)} 个共识，"
            f"{len(plan.disagreement_points)} 个分歧裁决。"
        )

        return plan

    def _parse_plan(self, content: str) -> ExecutionPlan:
        """从模型输出中解析执行计划"""
        plan = ExecutionPlan()

        # 提取共识点（## 共识分析 和 ## 分歧与裁决 之间的内容）
        consensus_section = self._extract_section(content, "## 共识分析", "## 分歧与裁决")
        if consensus_section:
            plan.consensus_points = [
                line.strip("- ").strip()
                for line in consensus_section.strip().split("\n")
                if line.strip().startswith("-")
            ]

        # 提取分歧与裁决
        disagreement_section = self._extract_section(content, "## 分歧与裁决", "## 执行计划")
        if disagreement_section:
            lines = disagreement_section.strip().split("\n")
            for line in lines:
                if line.strip().startswith("-"):
                    plan.disagreement_points.append({
                        "raw": line.strip("- ").strip(),
                    })

        # 提取 JSON 任务列表
        json_str = self._extract_json(content)
        if json_str:
            try:
                tasks_data = json.loads(json_str)
                if isinstance(tasks_data, list):
                    for td in tasks_data:
                        plan.add_task(Task(
                            title=td.get("title", "未命名任务"),
                            description=td.get("description", ""),
                            assigned_model=td.get("assigned_model", ""),
                            dependencies=td.get("dependencies", []),
                        ))
            except json.JSONDecodeError:
                # JSON 解析失败，尝试从 ```json 块中提取
                plan.rationale = "**注意**: 任务列表 JSON 解析失败，请检查主模型输出格式"
                # 兜底：把所有内容当作文本存储
                plan.rationale += f"\n\n原始输出:\n{content}"

        # 提取 rationale（如果有 ## 执行计划 之外的说明）
        plan.rationale = content

        return plan

    @staticmethod
    def _extract_section(text: str, header: str, next_header: str) -> str:
        """提取两个标题之间的文本"""
        start = text.find(header)
        if start == -1:
            return ""
        start += len(header)
        end = text.find(next_header, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从文本中提取 JSON 数组"""
        # 先尝试 ```json ... ``` 块
        import re
        match = re.search(r"```(?:json)?\s*\n?(\[.*?\])\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # 再尝试直接找 [ ] 包围的顶层 JSON
        start = text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None
