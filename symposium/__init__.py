"""
Symposium - 多模型研讨型协同 Agent 架构

核心理念：一组模型专家围绕任务进行圆桌研讨，达成共识后由主模型汇总决策，
然后协同执行，执行中通过消息总线实时沟通。

三阶段流程：
  1. Roundtable Deliberation - 多模型多轮研讨，互相启发
  2. Main Model Synthesis   - 主模型汇总共识、裁决分歧、产出执行计划
  3. Collaborative Execution - 按计划执行，任务间通过消息总线实时沟通
"""

__version__ = "0.1.0"
