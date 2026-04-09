"""
SYNTHEX model adapters — plug in any LLM backend.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseAdapter(ABC):
    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]: ...

    @abstractmethod
    def complete_sync(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models."""
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, max_tokens: int = 512):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        msg = await self._async_client.messages.create(
            model=self._model, max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return text.strip(), msg.usage.input_tokens + msg.usage.output_tokens

    def complete_sync(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        msg = self._client.messages.create(
            model=self._model, max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return text.strip(), msg.usage.input_tokens + msg.usage.output_tokens


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI GPT models."""
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, max_tokens: int = 512):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._async_client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        resp = await self._async_client.chat.completions.create(
            model=self._model, max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip(), resp.usage.total_tokens if resp.usage else 0

    def complete_sync(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        resp = self._client.chat.completions.create(
            model=self._model, max_tokens=max_tokens or self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip(), resp.usage.total_tokens if resp.usage else 0


class OllamaAdapter(BaseAdapter):
    """Adapter for local models via Ollama."""
    DEFAULT_MODEL = "llama3"
    DEFAULT_HOST  = "http://localhost:11434"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        try:
            import ollama as _ol
        except ImportError:
            raise ImportError("pip install ollama")
        self._model = model
        self._client = _ol.Client(host=host)
        self._async_client = _ol.AsyncClient(host=host)

    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}"

    async def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        resp = await self._async_client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip(), resp.get("eval_count", 0)

    def complete_sync(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        resp = self._client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip(), resp.get("eval_count", 0)


def create_adapter(model: str, **kwargs) -> BaseAdapter:
    """Auto-detect provider from model string."""
    provider = kwargs.pop("provider", None)
    if provider == "ollama" or model.startswith("ollama/"):
        return OllamaAdapter(model=model.replace("ollama/", ""), **kwargs)
    if provider == "openai" or model.startswith(("gpt-", "o1", "o3")):
        return OpenAIAdapter(model=model, **kwargs)
    if provider == "anthropic" or model.startswith("claude"):
        return AnthropicAdapter(model=model, **kwargs)
    raise ValueError(f"Cannot infer provider for '{model}'. Pass provider='anthropic'|'openai'|'ollama'.")
