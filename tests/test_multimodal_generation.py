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

import pytest
from pydantic import ValidationError

from app.config import settings
from app.multimodal import ResponsesRequest


def request_data(**overrides: object) -> dict[str, object]:
    return {
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }],
        **overrides,
    }


def test_responses_request_uses_safe_generation_defaults() -> None:
    request = ResponsesRequest.model_validate(request_data())

    assert request.generation.max_new_tokens == 1024
    assert request.generation.temperature is None
    assert request.generation.top_p == 0.9
    assert request.generation.top_k == 50
    assert request.generation.repetition_penalty == 1.1
    assert request.generation.truncation is True
    assert request.response_format == "text"


def test_responses_request_accepts_full_sampling_controls() -> None:
    request = ResponsesRequest.model_validate(request_data(
        generation={
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "truncation": False,
        }
    ))

    assert request.generation.top_p == 0.8
    assert request.generation.top_k == 20
    assert request.generation.repetition_penalty == 1.2
    assert request.generation.truncation is False


@pytest.mark.parametrize("generation", [
    {"top_p": 0.0},
    {"top_p": 1.1},
    {"top_k": -1},
    {"repetition_penalty": 0.0},
    {"repetition_penalty": 2.1},
])
def test_responses_request_rejects_unsafe_sampling_controls(
    generation: dict,
) -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(request_data(generation=generation))


def test_responses_request_accepts_json_object_format_and_temperature() -> None:
    request = ResponsesRequest.model_validate(request_data(
        generation={"max_new_tokens": 14000, "temperature": 0.3},
        response_format="json_object",
    ))

    assert request.generation.max_new_tokens == 14000
    assert request.generation.temperature == 0.3
    assert request.response_format == "json_object"


def test_responses_request_flags_default_to_off() -> None:
    request = ResponsesRequest.model_validate(request_data())

    assert request.retrieval is False
    assert request.child_friendly is False


def test_responses_request_rejects_child_friendly_json_output() -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(request_data(
            child_friendly=True,
            response_format="json_object",
        ))


def test_responses_request_rejects_unknown_response_format() -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(request_data(response_format="yaml"))


@pytest.mark.parametrize("max_new_tokens", [0, settings.MAX_RESPONSE_OUTPUT_TOKENS + 1])
def test_responses_request_rejects_unsafe_output_token_limit(
    max_new_tokens: int,
) -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(request_data(
            generation={"max_new_tokens": max_new_tokens}
        ))


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_responses_request_rejects_unsafe_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(request_data(
            generation={"temperature": temperature}
        ))
