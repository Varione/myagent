"""
远程 API 模型包装器 - OpenAI / 兼容 API 格式
"""

import json
import time
from typing import Optional

import requests

from .base import BaseModel, ChatMessage, ModelResponse


class RemoteModel(BaseModel):
    """远程 API 模型（OpenAI-compatible）"""

    def __init__(
        self,
        name: str,
        endpoint: str,
        api_key: str,
        model_name: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout: int = 180,
    ):
        self._name = name
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": False,
        }
        payload.update(kwargs)

        t0 = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
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
        return f"RemoteModel({self._name})"
