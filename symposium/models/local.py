"""
本地模型包装器 - 兼容 OpenAI API 格式的本地推理端点
"""

import json
import time
from typing import Optional

import requests

from .base import BaseModel, ChatMessage, ModelResponse


class LocalModel(BaseModel):
    """本地部署的 OpenAI-compatible 模型"""

    def __init__(
        self,
        name: str,
        endpoint: str,
        model_name: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        self._name = name
        self._endpoint = endpoint.rstrip("/")
        self._model_name = model_name or name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        url = f"{self._endpoint}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": False,
        }
        payload.update(kwargs)

        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=self._timeout)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]

        return ModelResponse(
            content=content,
            model_name=self._name,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={"elapsed_seconds": round(elapsed, 2), **(data.get("usage", {}))},
        )

    def __repr__(self) -> str:
        return f"LocalModel({self._name})"
