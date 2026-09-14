"""Model backend for remote OpenAI-compatible chat-completions endpoints."""
from __future__ import annotations

import json
import os
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models.base_model import BaseModel


class OpenAICompatibleModel(BaseModel):
    """Call a remote model without storing model weights in this repository."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model_id = config["model"]
        self.base_url = config["base_url"].rstrip("/")
        self.timeout = int(config.get("timeout", 120))
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 512))
        api_key_env = config.get("api_key_env")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    def generate(self, prompt: str, **kwargs) -> str:
        if "top_k" in kwargs:
            raise ValueError(
                "top_k is not part of the OpenAI-compatible API contract; "
                "use a provider-specific backend rather than silently changing "
                "the restricted-sampling condition"
            )
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get(
                "max_tokens", kwargs.get("max_new_tokens", self.max_tokens)
            ),
        }
        for key in ("top_p", "seed"):
            if key in kwargs:
                payload[key] = kwargs[key]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Remote generation failed for {self.name}: {exc}"
            ) from exc
        return body["choices"][0]["message"]["content"].strip()

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def sample_n(
        self, prompt: str, n: int = 5, temperature: float = 1.0
    ) -> List[str]:
        return [self.generate(prompt, temperature=temperature) for _ in range(n)]
