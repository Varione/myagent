"""Unified model adapter layer for local, remote, and mock models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import time
from typing import Optional

import requests


def _extract_error_message(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            error = data.get("error", {})
            if isinstance(error, dict):
                message = error.get("message")
                code = error.get("code")
                if message and code:
                    return f"{message} (code={code})"
                if message:
                    return message
        except Exception:
            try:
                text = response.text.strip()
                if text:
                    return text[:600]
            except Exception:
                pass
    return str(exc)


def _normalize_model_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


@dataclass
class ChatMessage:
    role: str = "user"
    content: str = ""

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelResponse:
    content: str = ""
    model_name: str = ""
    token_usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "success"
    error_message: str = ""
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class BaseModel(ABC):
    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ModelAdapter:
    """Registry and dispatcher for model backends."""

    def __init__(self):
        self._models: dict[str, BaseModel] = {}

    def register(self, name: str, model: BaseModel):
        self._models[name] = model

    def get(self, name: str) -> Optional[BaseModel]:
        return self._models.get(name)

    def chat(self, model_name: str, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        model = self._models.get(model_name)
        if model is None:
            message = f"错误: 模型 '{model_name}' 未注册"
            return ModelResponse(
                content=message,
                model_name=model_name,
                status="error",
                error_message=message,
            )
        try:
            return model.chat(messages, **kwargs)
        except Exception as exc:
            error_message = _extract_error_message(exc)
            return ModelResponse(
                content="",
                model_name=model_name,
                status="error",
                error_message=error_message,
                metadata={"exception_type": exc.__class__.__name__},
            )

    @property
    def available_models(self) -> list[str]:
        return list(self._models.keys())

    def __repr__(self) -> str:
        return f"ModelAdapter(models={self.available_models})"


class LocalModel(BaseModel):
    """OpenAI-compatible local endpoint."""

    def __init__(self, name: str, endpoint: str, model_name: str = "", timeout: int = 120):
        self._name = name
        self._endpoint = endpoint.rstrip("/")
        self._model_name = model_name or name
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def configured_model_id(self) -> str:
        return self._model_name

    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        return self._request_chat(messages, kwargs, allow_fallback=True)

    def _request_chat(self, messages: list[ChatMessage], kwargs: dict, allow_fallback: bool) -> ModelResponse:
        t0 = time.time()
        payload = {
            "model": self._model_name,
            "messages": [message.to_dict() for message in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{self._endpoint}/chat/completions",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - t0) * 1000
            return ModelResponse(
                content=content,
                model_name=self._name,
                token_usage=data.get("usage", {}),
                latency_ms=round(elapsed, 1),
                metadata={"resolved_model_id": self._model_name},
            )
        except Exception as exc:
            if allow_fallback:
                fallback_id = self._resolve_loaded_model_id()
                if fallback_id and fallback_id != self._model_name:
                    original = self._model_name
                    self._model_name = fallback_id
                    retry = self._request_chat(messages, kwargs, allow_fallback=False)
                    if retry.status != "error":
                        retry.metadata["configured_model_id"] = original
                        retry.metadata["resolved_model_id"] = fallback_id
                        retry.metadata["model_id_auto_corrected"] = True
                        return retry
                    self._model_name = original
            elapsed = (time.time() - t0) * 1000
            error_message = _extract_error_message(exc)
            available = self._fetch_available_model_ids()
            metadata = {
                "configured_model_id": self._model_name,
                "available_model_ids": available,
                "endpoint": self._endpoint,
            }
            return ModelResponse(
                content="",
                model_name=self._name,
                status="error",
                latency_ms=round(elapsed, 1),
                error_message=error_message,
                metadata=metadata,
            )

    def _resolve_loaded_model_id(self) -> str:
        available = self._fetch_available_model_ids()
        wanted = _normalize_model_key(self._model_name)
        for model_id in available:
            if _normalize_model_key(model_id) == wanted:
                return model_id
        return ""

    def _fetch_available_model_ids(self) -> list[str]:
        try:
            resp = requests.get(f"{self._endpoint}/models", timeout=min(self._timeout, 15))
            resp.raise_for_status()
            data = resp.json()
            return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
        except Exception:
            return []


class RemoteModel(BaseModel):
    """OpenAI-compatible remote endpoint."""

    def __init__(self, name: str, endpoint: str, api_key: str, model_name: str = "", timeout: int = 180):
        self._name = name
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name or name
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def configured_model_id(self) -> str:
        return self._model_name

    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self._endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [message.to_dict() for message in messages],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 8192),
                    "stream": False,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - t0) * 1000
            return ModelResponse(
                content=content,
                model_name=self._name,
                token_usage=data.get("usage", {}),
                latency_ms=round(elapsed, 1),
                metadata={"resolved_model_id": self._model_name},
            )
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            return ModelResponse(
                content="",
                model_name=self._name,
                status="error",
                latency_ms=round(elapsed, 1),
                error_message=_extract_error_message(exc),
                metadata={
                    "configured_model_id": self._model_name,
                    "endpoint": self._endpoint,
                },
            )


class MockModel(BaseModel):
    """Mock model used in tests and offline demos."""

    def __init__(self, name: str, responses: Optional[dict[str, str]] = None):
        self._name = name
        self._call_count = 0
        self._responses = responses or {}

    @property
    def name(self) -> str:
        return self._name

    def add_response(self, keyword: str, response: str):
        self._responses[keyword] = response

    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        self._call_count += 1
        full_text = "\n".join(message.content for message in messages)
        for keyword, response in self._responses.items():
            if keyword in full_text:
                return ModelResponse(content=response, model_name=self._name)
        return ModelResponse(
            content=f"[{self._name} 模拟响应] 已收到任务，执行完毕。调用次数: {self._call_count}",
            model_name=self._name,
        )

    @property
    def call_count(self) -> int:
        return self._call_count
