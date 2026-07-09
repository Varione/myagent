"""
协同执行器 - 按执行计划分配任务，模型间实时沟通

执行流程：
  1. 主模型持续监控进度
  2. 就绪任务分配给对应模型执行
  3. 执行中的模型可通过消息总线进行：
     - 进度同步
     - 问题求助（其他模型可响应）
     - 交叉验证
     - 实时修正
  4. 主模型可在执行中动态调整计划
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable

from ..models.base import BaseModel, ChatMessage
from .message_bus import (
    MessageBus,
    CHANNEL_PROGRESS,
    CHANNEL_PROBLEM,
    CHANNEL_SOLUTION,
    CHANNEL_CONTROL,
    CHANNEL_RESULT,
    CHANNEL_CROSS_REVIEW,
)
from .synthesizer import ExecutionPlan, Task, TaskStatus


# ──────────────────────────────────────────────
# 执行结果
# ──────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """整个执行阶段的结果"""
    task_results: dict[str, str] = field(default_factory=dict)  # task_id → output
    total_time: float = 0.0
    success: bool = True
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 任务执行感知器（每个任务运行时的上下文）
# ──────────────────────────────────────────────


class TaskContext:
    """任务执行上下文 - 让模型能感知总线动态并实时沟通"""

    def __init__(
        self,
        task: Task,
        model: BaseModel,
        bus: MessageBus,
        all_models: dict[str, BaseModel],
    ):
        self.task = task
        self.model = model
        self.bus = bus
        self.all_models = all_models

    def report_progress(self, message: str):
        """上报进度"""
        self.bus.publish(CHANNEL_PROGRESS, self.model.name, message,
                          metadata={"task_id": self.task.id})

    def ask_for_help(self, question: str) -> Optional[str]:
        """向其他模型求助（阻塞等待第一个回复）"""
        msg = self.bus.publish(CHANNEL_PROBLEM, self.model.name, question,
                                metadata={"task_id": self.task.id})

        # 等待最多 5 秒，看是否有其他模型回复
        deadline = time.time() + 5
        while time.time() < deadline:
            solutions = self.bus.get_channel_history(CHANNEL_SOLUTION)
            relevant = [
                s for s in solutions
                if s.metadata.get("in_reply_to") == msg.id
            ]
            if relevant:
                return relevant[-1].content
            time.sleep(0.5)

        return None

    def request_review(self, work_output: str) -> Optional[str]:
        """请求其他模型交叉验证"""
        msg = self.bus.publish(
            CHANNEL_CROSS_REVIEW,
            self.model.name,
            f"请审查以下输出:\n\n{work_output}",
            metadata={"task_id": self.task.id},
        )
        deadline = time.time() + 3
        while time.time() < deadline:
            reviews = self.bus.get_channel_history(CHANNEL_CROSS_REVIEW)
            relevant = [
                r for r in reviews
                if r.metadata.get("in_reply_to") == msg.id
            ]
            if relevant:
                return relevant[-1].content
            time.sleep(0.5)
        return None

    def get_recent_activity(self, max_messages: int = 20) -> str:
        """获取最近的全局动态（用于感知环境变化）"""
        return self.bus.format_channel_context(CHANNEL_PROGRESS, max_messages)


# ──────────────────────────────────────────────
# 协同执行器
# ──────────────────────────────────────────────


class CollaborativeExecutor:
    """协同执行器"""

    def __init__(
        self,
        plan: ExecutionPlan,
        models: dict[str, BaseModel],
        message_bus: MessageBus,
        main_model: Optional[BaseModel] = None,
        max_workers: int = 3,
        progress_callback: Optional[Callable] = None,
    ):
        self.plan = plan
        self.models = models
        self.bus = message_bus
        self.main_model = main_model
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self._stop_flag = threading.Event()

        # 启动一个后台线程来监听求助频道并自动分配合适的模型响应
        self._helper_thread = None

    # ── 执行入口 ──

    def execute(self) -> ExecutionResult:
        """执行全部任务（按依赖顺序）"""
        t0 = time.time()
        result = ExecutionResult()

        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"开始协同执行: {len(self.plan.tasks)} 个任务, "
            f"最大并行数 {self.max_workers}",
        )

        # 启动帮助线程（自动响应求助）
        self._start_helper()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}

            while not self.plan.all_done() and not self._stop_flag.is_set():
                ready_tasks = self.plan.get_ready_tasks()
                for task in ready_tasks:
                    if task.id in futures:
                        continue
                    model = self.models.get(task.assigned_model)
                    if model is None:
                        result.errors.append(
                            f"任务 '{task.title}': 模型 '{task.assigned_model}' 未注册"
                        )
                        task.status = TaskStatus.FAILED
                        continue

                    task.status = TaskStatus.RUNNING
                    ctx = TaskContext(task, model, self.bus, self.models)
                    future = pool.submit(self._execute_task, ctx, task)
                    futures[future] = task

                    self.bus.publish(
                        CHANNEL_PROGRESS,
                        "★ 调度器",
                        f"任务 '{task.title}' → {model.name}",
                    )

                # 收集已完成的任务
                done_futures = {f for f in futures if f.done()}
                for f in done_futures:
                    task = futures[f]
                    try:
                        output = f.result()
                        task.status = TaskStatus.DONE
                        task.output = output
                        result.task_results[task.id] = output
                        self.bus.publish(
                            CHANNEL_RESULT,
                            task.assigned_model,
                            f"任务 '{task.title}' 完成",
                            metadata={"task_id": task.id},
                        )
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        result.errors.append(f"任务 '{task.title}' 失败: {e}")
                        result.success = False
                        self.bus.publish(
                            CHANNEL_PROBLEM,
                            "★ 系统",
                            f"任务 '{task.title}' 执行失败: {e}",
                        )
                    del futures[f]

                if not ready_tasks and not futures:
                    # 没有就绪任务也没有运行中的任务
                    blocked = [t for t in self.plan.tasks if t.status == TaskStatus.PENDING]
                    if blocked:
                        result.errors.append(
                            f"死锁检测: {len(blocked)} 个任务因依赖无法启动"
                        )
                        result.success = False
                    break

                time.sleep(0.2)  # 避免忙等

            # 取消未完成的 future
            for f in futures:
                f.cancel()

        result.total_time = time.time() - t0

        # 汇总
        done_count = sum(1 for t in self.plan.tasks if t.status == TaskStatus.DONE)
        failed_count = sum(1 for t in self.plan.tasks if t.status == TaskStatus.FAILED)
        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"执行完毕: {done_count}/{len(self.plan.tasks)} 完成, "
            f"{failed_count} 失败, 耗时 {result.total_time:.1f}s",
        )

        self._stop_helper()
        return result

    # ── 单个任务执行 ──

    def _execute_task(self, ctx: TaskContext, task: Task) -> str:
        """执行单个任务（模型可在此过程中通过 ctx 实时沟通）"""
        system_prompt = (
            "你是 Symposium 协同执行中的一名专家。"
            "你的任务是完成分配给你的子任务。"
            "你可以随时通过 report_progress() 同步进度，"
            "通过 ask_for_help() 向其他专家求助，"
            "通过 request_review() 请求交叉验证。"
            "输出应该是可直接交付的成果。"
        )

        # 获取最近的全局动态（其他任务的进度、决策变更等）
        recent_context = ctx.get_recent_activity(10)

        user_prompt = (
            f"## 任务\n\n{task.title}\n\n"
            f"## 任务描述\n\n{task.description}\n\n"
            f"## 近期动态\n\n{recent_context}\n\n"
            "请执行此任务并输出结果。如果需要帮助或需要同步进度，在输出中标注。"
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        ctx.report_progress(f"开始执行: {task.title}")

        try:
            response = ctx.model.chat(messages)
            output = response.content.strip()

            ctx.report_progress(f"完成: {task.title}")
            return output
        except Exception as e:
            ctx.report_progress(f"失败: {task.title} - {e}")
            raise

    # ── 帮助线程（自动响应求助） ──

    def _start_helper(self):
        """启动后台线程，自动分配其他模型响应求助"""
        def helper_loop():
            answered = set()
            while not self._stop_flag.is_set():
                problems = self.bus.get_channel_history(CHANNEL_PROBLEM)
                for prob in problems:
                    if prob.id in answered:
                        continue
                    # 找到一个非求助者自身的模型来回答
                    for name, model in self.models.items():
                        if name == prob.sender:
                            continue
                        # 回复求助
                        reply = self._answer_question(model, prob.content)
                        self.bus.publish(
                            CHANNEL_SOLUTION,
                            model.name,
                            reply,
                            metadata={"in_reply_to": prob.id},
                        )
                        answered.add(prob.id)
                        break
                time.sleep(1.0)

        self._helper_thread = threading.Thread(target=helper_loop, daemon=True)
        self._helper_thread.start()

    def _answer_question(self, model: BaseModel, question: str) -> str:
        """让模型回答问题"""
        messages = [
            ChatMessage(
                role="system",
                content="你是 Symposium 协作团队中的一名专家。"
                        "另一名专家在执行任务时遇到了问题，请提供帮助。"
                        "回答要具体、可操作。"
            ),
            ChatMessage(role="user", content=f"求助问题:\n\n{question}"),
        ]
        try:
            resp = model.chat(messages, max_tokens=1024)
            return resp.content.strip()
        except Exception:
            return "（未能生成回复）"

    def _stop_helper(self):
        self._stop_flag.set()
        self._helper_thread = None

    def stop(self):
        """手动停止执行"""
        self._stop_flag.set()
