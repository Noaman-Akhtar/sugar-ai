# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for provider helper methods."""

from app.providers.base import BaseProvider, GenerationParams
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider


def test_openai_compatible_options_use_standard_fields():
    provider = object.__new__(BaseProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
        repetition_penalty=1.3,
    )

    options = provider._params_to_options(params)

    assert options == {
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.8,
    }


def test_ollama_options_use_ollama_field_names():
    provider = object.__new__(OllamaProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
        repetition_penalty=1.3,
    )

    options = provider._params_to_options(params)

    assert options == {
        "num_predict": 256,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 10,
        "repeat_penalty": 1.3,
    }


def test_gemini_config_uses_gemini_field_names():
    provider = object.__new__(GeminiProvider)
    params = GenerationParams(
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=10,
    )

    config = provider._params_to_config(params)

    assert config == {
        "maxOutputTokens": 256,
        "temperature": 0.2,
        "topP": 0.8,
        "topK": 10,
    }
