"""
Robust LLM controllers (plain-text, no JSON schema dependency).

Moved out of memory_layer_robust.py to keep the memory layer focused on memory logic.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Literal, Optional

logger = logging.getLogger("amem_robust")


def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                            e,
                            delay,
                        )
                        time.sleep(delay)
            logger.error(
                "LLM call %s failed after %d attempts: %s",
                func.__name__,
                max_retries + 1,
                last_exc,
            )
            raise last_exc

        return wrapper

    return decorator


class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion(
                "Reply with exactly one word: READY", temperature=0.0
            )
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info(
                "LLM connectivity check passed (response: %s)",
                response.strip()[:50],
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        self.model = model

        # If using DeepSeek via OpenAI-compatible API, auto-route by model prefix.
        # Example model names: "deepseek-chat", "deepseek-reasoner", etc.
        is_deepseek = str(model).lower().startswith("deepseek")
        if is_deepseek and api_base is None:
            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/")
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not found. Set DEEPSEEK_API_KEY (for deepseek*) or OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)
        # Expose last usage for prompt token length stats.
        self.last_prompt_tokens: Optional[int] = None

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        try:
            self.last_prompt_tokens = int(getattr(response, "usage").prompt_tokens)  # type: ignore[attr-defined]
        except Exception:
            self.last_prompt_tokens = None
        return response.choices[0].message.content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError(
                "ollama package not found. Install it with: pip install ollama"
            )
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "llama2",
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        import requests as _requests

        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            },
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(
            f"SGLang server returned status {response.status_code}: {response.text}"
        )


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(
        self,
        model: str = "llama2",
        vllm_host: str = "http://localhost",
        vllm_port: int = 30000,
    ):
        import requests as _requests

        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(
            f"vLLM server returned status {response.status_code}: {response.text}"
        )


class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(
        self,
        backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        check_connection: bool = False,
    ):
        if backend == "openai":
            # Auto-route DeepSeek models by prefix (deepseek*) to api_base.
            self.llm = RobustOpenAIController(model, api_key, api_base)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()

