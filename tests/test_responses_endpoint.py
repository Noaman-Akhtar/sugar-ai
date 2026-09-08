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

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.providers.base import ProviderResponse
from app.routes import api


API_KEY = "test-key"
PNG_DATA = base64.b64encode(b"\x89PNG\r\n\x1a\nsmall-png").decode("ascii")


class FakeProvider:
    def __init__(self, modalities=("text",), formats=("text",)):
        self.modalities = frozenset(modalities)
        self.formats = frozenset(formats)

    def supports_input_modalities(self, required_modalities) -> bool:
        return set(required_modalities) <= self.modalities

    def supports_response_format(self, response_format) -> bool:
        return response_format in self.formats


class FakeAgent:
    def __init__(self, provider, result=ProviderResponse("Hello!", "completed")):
        self.provider = provider
        self.result = result
        self.calls = []

    def run_multimodal(
        self, messages, params=None, response_format="text", retrieval=False
    ):
        self.calls.append((messages, params, response_format))
        self.retrieval_flags = getattr(self, "retrieval_flags", [])
        self.retrieval_flags.append(retrieval)
        return self.result


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        settings, "API_KEYS", {API_KEY: {"name": "tester"}}, raising=False
    )
    api.user_quotas.clear()
    app = FastAPI()
    app.include_router(api.router)
    yield TestClient(app)
    api.user_quotas.clear()


def install_agent(monkeypatch, agent: FakeAgent) -> FakeAgent:
    monkeypatch.setattr(api, "agent", agent)
    return agent


def post_responses(client, payload):
    return client.post("/v1/responses", json=payload, headers={"X-API-Key": API_KEY})


def text_payload(**overrides) -> dict:
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi there"}]}
        ]
    }
    payload.update(overrides)
    return payload


def image_payload(data: str = PNG_DATA) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "source": {"type": "base64", "data": data},
                    },
                ],
            }
        ]
    }


def test_text_only_request_succeeds(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = post_responses(client, text_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("resp_")
    assert body["status"] == "completed"
    assert body["output"] == [{"type": "text", "text": "Hello!"}]
    assert body["quota"] == {
        "used_units": 1,
        "remaining_units": settings.MAX_DAILY_REQUESTS - 1,
        "daily_limit_units": settings.MAX_DAILY_REQUESTS,
    }
    assert len(agent.calls) == 1


def test_image_request_succeeds_with_image_capable_provider(client, monkeypatch):
    agent = install_agent(
        monkeypatch, FakeAgent(FakeProvider(modalities=("text", "image")))
    )

    response = post_responses(client, image_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["quota"]["used_units"] == 3
    messages, _, _ = agent.calls[0]
    assert messages[0].role == "user"


def test_json_object_request_returns_json_output(client, monkeypatch):
    agent = FakeAgent(
        FakeProvider(formats=("text", "json_object")),
        result=ProviderResponse('{"animal": "cat"}', "completed"),
    )
    install_agent(monkeypatch, agent)

    response = post_responses(
        client, text_payload(response_format="json_object")
    )

    assert response.status_code == 200
    assert response.json()["output"] == [
        {"type": "json", "json": {"animal": "cat"}}
    ]
    assert agent.calls[0][2] == "json_object"


def test_image_to_text_only_provider_is_rejected_without_charge(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = post_responses(client, image_payload())

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_modality"
    assert agent.calls == []
    assert api.remaining_quota_units(API_KEY) == settings.MAX_DAILY_REQUESTS


def test_json_object_to_text_only_provider_is_rejected(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = post_responses(
        client, text_payload(response_format="json_object")
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_response_format"
    assert agent.calls == []


def test_invalid_base64_is_rejected_without_charge(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = post_responses(client, image_payload(data="not-base64!!"))

    assert response.status_code == 422
    assert agent.calls == []
    assert api.remaining_quota_units(API_KEY) == settings.MAX_DAILY_REQUESTS


def test_insufficient_quota_returns_429_before_provider_call(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))
    api.consume_quota_units(API_KEY, settings.MAX_DAILY_REQUESTS)

    response = post_responses(client, text_payload())

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "insufficient_quota"
    assert agent.calls == []


def test_incomplete_provider_result_is_reported_with_reason(client, monkeypatch):
    install_agent(
        monkeypatch,
        FakeAgent(FakeProvider(), result=ProviderResponse("Partial", "incomplete")),
    )

    response = post_responses(client, text_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_reason"] == "output_limit"
    assert body["output"] == [{"type": "text", "text": "Partial"}]


def test_malformed_provider_json_is_not_returned_as_structured_data(
    client, monkeypatch
):
    install_agent(
        monkeypatch,
        FakeAgent(
            FakeProvider(formats=("text", "json_object")),
            result=ProviderResponse("not json at all", "completed"),
        ),
    )

    response = post_responses(
        client, text_payload(response_format="json_object")
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_provider_output"


def test_sampling_controls_reach_the_provider(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = post_responses(client, text_payload(generation={
        "max_new_tokens": 200,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.2,
        "truncation": False,
    }))

    assert response.status_code == 200
    _, params, _ = agent.calls[0]
    assert params.max_new_tokens == 200
    assert params.temperature == 0.3
    assert params.top_p == 0.8
    assert params.top_k == 20
    assert params.repetition_penalty == 1.2
    assert params.truncation is False


def test_retrieval_flag_reaches_the_agent(client, monkeypatch):
    agent = install_agent(monkeypatch, FakeAgent(FakeProvider()))

    off = post_responses(client, text_payload())
    on = post_responses(client, text_payload(retrieval=True))

    assert off.status_code == 200
    assert on.status_code == 200
    assert agent.retrieval_flags == [False, True]


def test_missing_api_key_returns_401(client, monkeypatch):
    install_agent(monkeypatch, FakeAgent(FakeProvider()))

    response = client.post("/v1/responses", json=text_payload())

    assert response.status_code == 401
