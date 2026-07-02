# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""OpenAI-compatible chat completions provider for Sugar-AI.

Works with any service that implements the OpenAI /v1/chat/completions
format: Groq, Cerebras, Together.ai, OpenRouter, OpenAI, Mistral, etc.
The only differences between them are base_url, api_key, and model name.
"""
import httpx
import logging
from typing import Optional

from app.providers.base import BaseProvider, GenerationParams

logger = logging.getLogger("sugar-ai")

# Cloud APIs are usually fast, but allow headroom for cold routes / rate-limit
# retries handled upstream. 120s is generous without hanging forever.
_DEFAULT_TIMEOUT = 120.0


class OpenAICompatibleProvider(BaseProvider):
    """Provider for any OpenAI-format /v1/chat/completions endpoint.

    base_url must include the version segment, e.g.:
      - OpenAI:     https://api.openai.com/v1
      - Groq:       https://api.groq.com/openai/v1
      - Cerebras:   https://api.cerebras.ai/v1
      - OpenRouter: https://openrouter.ai/api/v1
      - Together:   https://api.together.xyz/v1
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        if not api_key:
            raise ValueError(
                "OpenAICompatibleProvider requires an api_key. "
                "Set OPENAI_API_KEY in your environment."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=_DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "OpenAICompatibleProvider initialized: model=%s, server=%s",
            model_name,
            self.base_url,
        )

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate text from a plain prompt by wrapping it as a user message."""
        return self.chat([{"role": "user", "content": prompt}], params)

    def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        """Generate a response from chat messages via /chat/completions."""
        if params is None:
            params = GenerationParams()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **self._params_to_options(params),
        }

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return (message.get("content") or "").strip()

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        """Verify the endpoint is reachable and the key/model are valid."""
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def _params_to_options(self, params: GenerationParams) -> dict:
        """Map GenerationParams to OpenAI chat-completions fields.

        Only OpenAI-standard fields are sent. top_k and repetition_penalty
        are not part of the spec and are intentionally omitted.
        """
        return {
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
