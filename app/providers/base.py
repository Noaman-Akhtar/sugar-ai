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


"""Base provider interface for Sugar-AI."""
import httpx
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from app.multimodal import NormalizedImage, NormalizedMessage, NormalizedText

logger = logging.getLogger("sugar-ai")

# Cloud APIs are usually fast, but allow headroom for cold routes / rate-limit
# retries handled upstream. 120s is generous without hanging forever.
_DEFAULT_TIMEOUT = 120.0

InputModality = Literal["text", "image"]
ResponseFormat = Literal["text", "json_object"]


class UnsupportedModalityError(ValueError):
    """Raised when a provider cannot accept required input content."""


class UnsupportedResponseFormatError(ValueError):
    """Raised when a provider cannot reliably produce a requested format."""


@dataclass(frozen=True)
class ProviderResponse:
    """Provider-neutral generated text and its completion state."""

    text: str
    status: Literal["completed", "incomplete"]


@dataclass(frozen=True)
class GenerationParams:
    """Parameters controlling text generation behavior."""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    truncation: bool = True
    do_sample: bool = True

    def __post_init__(self):
        object.__setattr__(self, "do_sample", self.temperature > 0)


class BaseProvider:
    """OpenAI-compatible provider: speaks /v1/chat/completions over HTTP."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        if not api_key:
            raise ValueError(
                f"{type(self).__name__} requires an api_key. "
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
            "%s initialized: model=%s, server=%s",
            type(self).__name__,
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

    def supported_input_modalities(self) -> frozenset[InputModality]:
        """Return input modalities this provider can handle safely."""
        return frozenset({"text"})

    def supports_input_modalities(
        self, required_modalities: set[InputModality]
    ) -> bool:
        """Return whether every required input modality is supported."""
        return required_modalities <= self.supported_input_modalities()

    def supports_response_format(self, response_format: ResponseFormat) -> bool:
        """Return whether this provider can reliably produce the format."""
        return response_format == "text"

    def generate_multimodal(
        self,
        messages: tuple[NormalizedMessage, ...],
        params: Optional[GenerationParams] = None,
        response_format: ResponseFormat = "text",
    ) -> ProviderResponse:
        """Generate from normalized messages in a provider-specific adapter."""
        required_modalities: set[InputModality] = {"text"}
        if any(
            isinstance(part, NormalizedImage)
            for message in messages
            for part in message.content
        ):
            required_modalities.add("image")

        if not self.supports_input_modalities(required_modalities):
            unsupported_modalities = required_modalities - self.supported_input_modalities()
            raise UnsupportedModalityError(
                f"{type(self).__name__} does not support "
                f"{', '.join(sorted(unsupported_modalities))} input"
            )
        if not self.supports_response_format(response_format):
            raise UnsupportedResponseFormatError(
                f"{type(self).__name__} does not support {response_format} responses"
            )
        if "image" in required_modalities:
            raise UnsupportedModalityError(
                f"{type(self).__name__} has not implemented image generation"
            )

        # Text-only fallback: any provider with a chat() method can serve
        # normalized text messages. chat() cannot report truncation, so the
        # result is marked completed, matching the legacy endpoints.
        chat_messages = [
            {
                "role": message.role,
                "content": "\n\n".join(
                    part.text
                    for part in message.content
                    if isinstance(part, NormalizedText)
                ),
            }
            for message in messages
        ]
        return ProviderResponse(
            text=self.chat(chat_messages, params),
            status="completed",
        )

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
        are not part of the spec and are intentionally omitted."""
        
        return {
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }

    def get_eos_token(self) -> Optional[str]:
        """Return the provider's EOS token string if one is known."""
        return None

    def close(self) -> None:
        """Release provider resources."""
        pass
