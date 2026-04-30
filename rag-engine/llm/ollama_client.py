"""
Ollama local LLM client — supports both single-shot and SSE streaming.
Thinking support: reasoning models (deepseek-r1, qwen3, gpt-oss) emit a separate
`thinking` field when `think: true` is set on the request. Other models error out
on that flag, so we gate it via `_supports_thinking()`.
"""
import json
import os
from collections.abc import Iterator
from urllib.parse import urlparse

import requests


# Resolve Ollama base URL from env (matches predev OLLAMA_URL + old project LOCAL_LLM_URL)
def _resolve_base_url() -> str:
    raw = os.getenv("OLLAMA_URL") or os.getenv("LOCAL_LLM_URL") or "http://localhost:11434"
    # Strip trailing path like /api/chat or /api/generate to keep just scheme://host:port
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw.rstrip("/")


OLLAMA_BASE_URL = _resolve_base_url()
DEFAULT_TIMEOUT = 120

# Model families that expose a separate `thinking` channel when `think: true`.
# Ollama returns 400 for non-thinking models, so we have to gate the flag.
_THINKING_MODEL_PREFIXES = ("deepseek-r1", "qwen3", "gpt-oss", "deepseek-v3")


def _supports_thinking(model: str) -> bool:
    name = (model or "").lower()
    return any(name.startswith(p) for p in _THINKING_MODEL_PREFIXES)


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str, model: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        """Single-shot generation — waits for full response."""
        url = f"{self._base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["response"]

    def generate_stream(
        self, prompt: str, model: str, timeout: int = DEFAULT_TIMEOUT
    ) -> Iterator[str]:
        """Streaming generation — yields response tokens one by one (no thinking)."""
        for ev in self.generate_stream_full(prompt, model=model, think=False, timeout=timeout):
            if ev["kind"] == "response":
                yield ev["token"]

    def generate_stream_full(
        self,
        prompt: str,
        model: str,
        think: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Iterator[dict]:
        """
        Streaming generation that yields BOTH thinking and response tokens
        as separate events:
          {"kind": "thinking", "token": "..."}
          {"kind": "response", "token": "..."}
        For non-reasoning models (or think=False), only "response" events are emitted.
        """
        url = f"{self._base_url}/api/generate"
        payload: dict = {"model": model, "prompt": prompt, "stream": True}
        # For reasoning models, send `think` explicitly (true OR false) — omitting it
        # makes Ollama default to thinking ON, so we MUST send `think: false` to disable.
        # Non-reasoning models 400 on the flag, so we skip it entirely there.
        if _supports_thinking(model):
            payload["think"] = bool(think)

        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                # Thinking comes first; once response starts the model is "answering"
                think_tok = data.get("thinking") or ""
                if think_tok:
                    yield {"kind": "thinking", "token": think_tok}
                resp_tok = data.get("response") or ""
                if resp_tok:
                    yield {"kind": "response", "token": resp_tok}
                if data.get("done"):
                    break

    def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        url = f"{self._base_url}/api/tags"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
