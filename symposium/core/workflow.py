"""
全流程编排入口 - 串联研讨 → 汇总 → 执行三个阶段

外部只需一行调用：
    workflow = SymposiumWorkflow(config)
    result = workflow.run("我的任务...")
"""

import time
from typing import Optional

from ..models.base import BaseModel
from ..models.local import LocalModel
from ..models.remote import RemoteModel
from ..config import SymposiumConfig, ModelConfig
from .message_bus import MessageBus, CHANNEL_CONTROL
from .deliberation import DeliberationEngine
from .synthesizer import MainModelSynthesizer
from .executor import CollaborativeExecutor, ExecutionResult


class SymposiumWorkflow:
    """全流程编排器"""

    def __init__(
        self,
        config: SymposiumConfig,
        message_bus: Optional[MessageBus] = None,
        models: Optional[dict[str, BaseModel]] = None,
    ):
        self.config = config
        self.bus = message_bus or MessageBus()
        self.models: dict[str, BaseModel] = models or {}

        if models is None:
            # 根据配置启动模型
            self._init_models()
            # 验证主模型
            if config.main_model and config.main_model not in self.models:
                raise ValueError(
                    f"主模型 '{config.main_model}' 未注册。可用模型: {list(self.models.keys())}"
                )
        elif config.main_model and config.main_model not in self.models:
            raise ValueError(
                f"注入的模型列表中缺少主模型 '{config.main_model}'。可用: {list(self.models.keys())}"
            )

    def _init_models(self):
        """根据配置初始化所有模型"""
        for name, cfg in self.config.models.items():
            model = self._build_model(cfg)
            self.models[name] = model

    @staticmethod
    def _build_model(cfg: ModelConfig) -> BaseModel:
        if cfg.provider == "local":
            return LocalModel(
                name=cfg.name,
                endpoint=cfg.endpoint,
                model_name=cfg.model_name or cfg.name,
                temperature=cfg.temperature,
            )
        elif cfg.provider == "remote":
            return RemoteModel(
                name=cfg.name,
                endpoint=cfg.endpoint,
                api_key=cfg.api_key,
                model_name=cfg.model_name or cfg.name,
                temperature=cfg.temperature,
            )
        else:
            raise ValueError(f"未知 provider: {cfg.provider}")

    def get_main_model(self) -> BaseModel:
        """获取主模型实例"""
        main_name = self.config.main_model
        model = self.models.get(main_name)
        if model is None:
            raise ValueError(f"主模型 '{main_name}' 未初始化")
        return model

    def get_deliberation_models(self) -> list[BaseModel]:
        """获取参与研讨的模型列表（排除主模型）"""
        main_name = self.config.main_model
        return [m for name, m in self.models.items() if name != main_name]

    # ── 全流程 ──

    def run(
        self,
        task: str,
        task_context: Optional[str] = None,
        num_rounds: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> dict:
        """
        执行完整三阶段流程

        参数:
            task: 任务描述
            task_context: 任务背景信息（可选）
            num_rounds: 研讨轮数（覆盖配置）
            max_workers: 最大并行任务数（覆盖配置）

        返回:
            {
                "task": str,
                "deliberation": DeliberationResult,
                "plan": ExecutionPlan,
                "execution": ExecutionResult,
                "total_time": float,
                "transcript": dict,  # 全部消息记录
            }
        """
        t0 = time.time()

        # ── 阶段一: 圆桌研讨 ──
        debate_models = self.get_deliberation_models()
        if len(debate_models) < 2:
            raise ValueError(
                f"研讨需要至少 2 个模型（目前除去主模型后只有 {len(debate_models)} 个）"
            )

        rounds = num_rounds or self.config.deliberation.num_rounds
        engine = DeliberationEngine(
            models=debate_models,
            task=task,
            num_rounds=rounds,
            message_bus=self.bus,
            task_context=task_context,
        )
        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"=== 阶段一: 圆桌研讨 ==="
        )
        deliberation_result = engine.run()

        # ── 阶段二: 主模型汇总 ──
        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"=== 阶段二: 主模型汇总 ==="
        )
        synthesizer = MainModelSynthesizer(
            main_model=self.get_main_model(),
            message_bus=self.bus,
        )
        plan = synthesizer.synthesize(deliberation_result)

        # ── 阶段三: 协同执行 ──
        self.bus.publish(
            CHANNEL_CONTROL,
            "★ 系统",
            f"=== 阶段三: 协同执行 ==="
        )
        executor = CollaborativeExecutor(
            plan=plan,
            models=self.models,
            message_bus=self.bus,
            main_model=self.get_main_model(),
            max_workers=max_workers or self.config.execution.max_concurrent_tasks,
        )
        execution_result = executor.execute()

        total_time = time.time() - t0

        # ── 组装结果 ──
        return {
            "task": task,
            "deliberation": deliberation_result,
            "plan": plan,
            "execution": execution_result,
            "total_time": total_time,
            "transcript": {
                "all_messages": self.bus.get_all_messages(),
                "stats": self.bus.stats(),
            },
        }

    # ── 实用方法 ──

    def print_report(self, result: dict):
        """打印完整执行报告"""
        deliberation = result["deliberation"]
        plan = result["plan"]
        execution = result["execution"]

        print("\n" + "=" * 70)
        print("  Symposium 执行报告")
        print("=" * 70)
        print(f"\n任务: {result['task']}")
        print(f"总耗时: {result['total_time']:.1f}s")
        print(f"总线消息: {result['transcript']['stats']['total_messages']} 条")

        print(f"\n── 研讨阶段 ──")
        print(f"  发言: {len(deliberation.contributions)} 次")
        print(f"  轮数: {deliberation.num_rounds}")
        print(f"  耗时: {deliberation.total_time:.1f}s")

        print(f"\n── 汇总阶段 ──")
        print(f"  共识点: {len(plan.consensus_points)}")
        print(f"  分歧裁决: {len(plan.disagreement_points)}")
        print(f"  任务分解: {len(plan.tasks)} 个任务")
        if plan.tasks:
            for t in plan.tasks:
                deps = f" (依赖: {t.dependencies})" if t.dependencies else ""
                print(f"    {t.id}: {t.title} → {t.assigned_model}{deps}")

        print(f"\n── 执行阶段 ──")
        print(f"  完成: {sum(1 for t in plan.tasks if t.status.value == 'done')}/{len(plan.tasks)}")
        print(f"  失败: {sum(1 for t in plan.tasks if t.status.value == 'failed')}/{len(plan.tasks)}")
        print(f"  耗时: {execution.total_time:.1f}s")
        if execution.errors:
            print(f"  错误:")
            for e in execution.errors:
                print(f"    ⚠ {e}")

        print(f"\n── 完整研讨记录 ──")
        self.bus.print_channel("debate")

        print(f"\n── 控制/汇总记录 ──")
        self.bus.print_channel("control")

        print(f"\n── 执行进度 ──")
        self.bus.print_channel("progress")

        print(f"\n── 求助/解答 ──")
        self.bus.print_channel("problem")
        self.bus.print_channel("solution")
