"""
RoleRunner — 角色执行器

根据角色类型加载对应的 system prompt，结合 Task Contract 调用 ModelAdapter 执行。
"""

from typing import Optional

from ..models.adapter import ModelAdapter, ChatMessage, ModelResponse
from ..schemas.task_contract import TaskContract
from ..schemas.agent_response import AgentResponse
from ..schemas.blackboard_entry import BlackboardEntry
from .prompts import RolePromptLibrary


class RoleRunner:
    """角色执行器：一个类处理所有角色，通过 prompt template 区分"""

    def __init__(self, model_adapter: ModelAdapter):
        self.adapter = model_adapter
        self.prompts = RolePromptLibrary()

    def run(
        self,
        contract: TaskContract,
        model_name: str,
        context_messages: Optional[list[ChatMessage]] = None,
    ) -> AgentResponse:
        """
        执行角色任务。

        Args:
            contract: 子任务合同
            model_name: 使用的模型名称
            context_messages: 额外的上下文消息（如黑板条目）
        """
        system_prompt = self.prompts.get_prompt(contract.role)
        if not system_prompt:
            return AgentResponse(
                task_id=contract.task_id,
                role=contract.role,
                status="failed",
                summary=f"未知角色: {contract.role}",
            )

        # 构建消息列表
        messages = [ChatMessage(role="system", content=system_prompt)]

        # 添加上下文
        if context_messages:
            messages.extend(context_messages)

        # 添加任务合同
        user_message = (
            f"## Task Contract\n\n"
            f"Task ID: {contract.task_id}\n"
            f"任务名称: {contract.task_name}\n"
            f"目标: {contract.objective}\n"
            f"成功标准: {chr(10).join(f'- {s}' for s in contract.success_criteria)}\n"
            f"超时时间: {contract.timeout_seconds}秒\n"
        )
        if contract.input_refs:
            user_message += f"\n输入引用: {', '.join(contract.input_refs)}\n"

        user_message += (
            "\n请按你的角色要求完成任务，并按指定格式输出 JSON。"
        )
        messages.append(ChatMessage(role="user", content=user_message))

        # 调用模型
        resp = self.adapter.chat(model_name, messages)

        if resp.status == "error":
            return AgentResponse(
                task_id=contract.task_id,
                role=contract.role,
                status="failed",
                summary=f"模型调用失败: {resp.content}",
            )

        # 解析输出
        return self._parse_response(contract, resp)

    def _parse_response(self, contract: TaskContract, resp: ModelResponse) -> AgentResponse:
        """从模型原始输出中解析 AgentResponse"""
        content = resp.content.strip()

        # 尝试提取 JSON（模型可能输出 markdown 代码块）
        json_str = self._extract_json(content)

        if json_str:
            import json
            try:
                data = json.loads(json_str)
                return AgentResponse.from_dict({
                    "task_id": contract.task_id,
                    "role": contract.role,
                    "status": data.get("status", "completed"),
                    "summary": data.get("summary", ""),
                    "content": data.get("content", content),
                    "claims": data.get("claims", []),
                    "risks": data.get("risks", []),
                    "next_action_recommendation": data.get("next_action_recommendation", ""),
                })
            except json.JSONDecodeError:
                pass

        # 兜底：整段内容作为 summary
        return AgentResponse(
            task_id=contract.task_id,
            role=contract.role,
            status="completed",
            summary=content[:200],
            content=content,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从文本中提取 JSON 对象"""
        import re
        # 尝试 ```json ... ``` 格式
        match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # 直接找 {} 包围的 JSON
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None

    @staticmethod
    def blackboard_entries_to_context(entries: list[BlackboardEntry]) -> list[ChatMessage]:
        """将黑板条目转换为上下文消息"""
        if not entries:
            return []
        context = ""
        for entry in entries:
            context += (
                f"[{entry.content_type}] 来自 {entry.source_role}\n"
                f"{entry.summary}\n\n"
            )
        return [ChatMessage(role="user", content=f"## 上下文输入\n\n{context}")]
