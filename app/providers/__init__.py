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


"""
Provider package for Sugar-AI.

Providers are the abstraction layer between RAGAgent and model backends.
"""
from app.providers.base import BaseProvider, GenerationParams
from app.providers.huggingface import HuggingFaceProvider
from app.providers.ollama import OllamaProvider

__all__ = [
    "BaseProvider",
    "GenerationParams",
    "HuggingFaceProvider",
    "OllamaProvider",
    "create_provider",
]


def create_provider(
    provider_name: str,
    model_name: str,
    **kwargs,
) -> BaseProvider:
    """Build a configured model provider by name."""
    name = provider_name.lower().strip()

    if name == "huggingface":
        return HuggingFaceProvider(
            model_name=model_name,
            quantize=kwargs.get("quantize", True),
            dev_mode=kwargs.get("dev_mode", False),
        )

    if name == "ollama":
        return OllamaProvider(
            model_name=model_name,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )

    raise ValueError(
        f"Unknown provider: '{provider_name}'. "
        f"Valid providers: huggingface, ollama"
    )
