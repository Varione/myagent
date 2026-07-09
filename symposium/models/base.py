"""
模型抽象接口 - 所有模型包装器的基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """消息体（兼容 OpenAI 格式）"""
    role: str       # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelResponse:
    """结构化模型输出"""
    content: str
    model_name: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class BaseModel(ABC):
    """模型接口抽象"""

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        """发送对话消息，返回模型响应"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """模型唯一标识"""
        ...

    @property
    def config(self):
        """模型配置（由子类设置）"""
        return getattr(self, "_config", None)

    def build_system_prompt(self, persona: str) -> ChatMessage:
        """快捷构建系统提示"""
        return ChatMessage(role="system", content=persona)

    def build_user_message(self, content: str) -> ChatMessage:
        return ChatMessage(role="user", content=content)
