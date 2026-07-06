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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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


class BaseProvider(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        ...

    @abstractmethod
    def chat(self, messages: list[dict], params: Optional[GenerationParams] = None) -> str:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...

    def get_eos_token(self) -> Optional[str]:
        """Return the provider's EOS token string if one is known."""
        return None

    def close(self) -> None:
        """Release provider resources."""
        pass
