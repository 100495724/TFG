"""
models.py - Unified interface for calling LLMs.

Supports:
- vLLM servers (local on RunPod) via OpenAI-compatible API
- Anthropic API (Claude)
- OpenAI API (GPT)

All models expose the same `generate(messages) -> str` interface.
"""

import json
import os
import time
import logging
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all LLM backends."""

    def __init__(self, model_name: str, params: dict):
        self.model_name = model_name
        self.params = params

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a response given a system prompt and user message."""
        pass

    def _retry_with_backoff(self, func, max_retries=3, base_delay=5):
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)


class VLLMModel(BaseLLM):
    """Interface for vLLM servers running on RunPod (OpenAI-compatible API)."""

    def __init__(self, model_name: str, base_url: str, params: dict):
        super().__init__(model_name, params)
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

    def generate(self, system_prompt: str, user_message: str) -> str:
        def _call():
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.params.get("temperature", 0.3),
                "max_tokens": self.params.get("max_tokens", 1024),
                "top_p": self.params.get("top_p", 0.95),
            }
            # vLLM supports seed for reproducibility
            if "seed" in self.params:
                payload["seed"] = self.params["seed"]

            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        return self._retry_with_backoff(_call)


class AnthropicModel(BaseLLM):
    """Interface for Anthropic API (Claude models)."""

    def __init__(self, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.endpoint = "https://api.anthropic.com/v1/messages"

    def generate(self, system_prompt: str, user_message: str) -> str:
        def _call():
            payload = {
                "model": self.model_name,
                "max_tokens": self.params.get("max_tokens", 1024),
                "temperature": self.params.get("temperature", 0.3),
                "top_p": self.params.get("top_p", 0.95),
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_message},
                ],
            }
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

        return self._retry_with_backoff(_call)


class OpenAIModel(BaseLLM):
    """Interface for OpenAI API (GPT models)."""

    def __init__(self, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(self, system_prompt: str, user_message: str) -> str:
        def _call():
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.params.get("temperature", 0.3),
                "max_tokens": self.params.get("max_tokens", 1024),
                "top_p": self.params.get("top_p", 0.95),
            }
            if "seed" in self.params:
                payload["seed"] = self.params["seed"]

            response = requests.post(
                self.endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        return self._retry_with_backoff(_call)


def create_model(model_id: str, endpoints_config: dict, inference_params: dict) -> BaseLLM:
    """Factory function: creates the right LLM backend based on config."""
    cfg = endpoints_config[model_id]
    model_type = cfg["type"]

    if model_type == "vllm":
        return VLLMModel(
            model_name=cfg["model_name"],
            base_url=cfg["base_url"],
            params=inference_params,
        )
    elif model_type == "anthropic":
        return AnthropicModel(
            model_name=cfg["model_name"],
            params=inference_params,
        )
    elif model_type == "openai":
        return OpenAIModel(
            model_name=cfg["model_name"],
            params=inference_params,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
