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
import json

import httpx

from app.multimodal import NormalizedImage, NormalizedMessage, NormalizedText
from app.providers.gemini import GeminiProvider


def provider() -> GeminiProvider:
    return GeminiProvider(model_name="test-model", api_key="test-key")


def mocked_provider(response_data: dict, captured_request: dict) -> GeminiProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(request.content)
        return httpx.Response(200, json=response_data)

    gemini_provider = provider()
    gemini_provider._client.close()
    gemini_provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return gemini_provider


def test_normalized_messages_become_ordered_gemini_text_and_image_parts() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\npreview"
    messages = (
        NormalizedMessage(
            role="system",
            content=(NormalizedText(text="Be helpful."),),
        ),
        NormalizedMessage(
            role="user",
            content=(
                NormalizedText(text="Describe this."),
                NormalizedImage(
                    data=image_bytes,
                    media_type="image/png",
                    label="entry-preview",
                ),
                NormalizedText(text="Use simple words."),
            ),
        ),
        NormalizedMessage(
            role="assistant",
            content=(NormalizedText(text="Earlier reply."),),
        ),
    )

    contents, system_instruction = provider()._normalized_to_gemini_contents(messages)

    assert system_instruction == "Be helpful."
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "Describe this."},
                {"inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
                {"text": "Use simple words."},
            ],
        },
        {"role": "model", "parts": [{"text": "Earlier reply."}]},
    ]


def test_gemini_declares_image_and_json_support() -> None:
    gemini_provider = provider()

    assert gemini_provider.supported_input_modalities() == frozenset({"text", "image"})
    assert gemini_provider.supports_response_format("json_object")


def test_gemini_sends_images_and_json_output_request() -> None:
    captured_request: dict = {}
    gemini_provider = mocked_provider({
        "candidates": [{
            "content": {"parts": [{"text": "{\"answer\": \"A tree\"}"}]},
            "finishReason": "STOP",
        }],
    }, captured_request)
    image_bytes = b"\x89PNG\r\n\x1a\npreview"
    messages = (NormalizedMessage(
        role="user",
        content=(
            NormalizedText(text="What is this?"),
            NormalizedImage(
                data=image_bytes,
                media_type="image/png",
                label="preview",
            ),
        ),
    ),)

    response = gemini_provider.generate_multimodal(
        messages,
        response_format="json_object",
    )

    assert captured_request["url"].endswith(
        "/models/test-model:generateContent"
    )
    assert captured_request["payload"]["contents"][0]["parts"] == [
        {"text": "What is this?"},
        {"inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }},
    ]
    assert (
        captured_request["payload"]["generationConfig"]["responseMimeType"]
        == "application/json"
    )
    assert response.text == '{"answer": "A tree"}'
    assert response.status == "completed"


def test_gemini_marks_output_limit_response_as_incomplete() -> None:
    captured_request: dict = {}
    gemini_provider = mocked_provider({
        "candidates": [{
            "content": {"parts": [{"text": "Partial answer"}]},
            "finishReason": "MAX_TOKENS",
        }],
    }, captured_request)
    messages = (NormalizedMessage(
        role="user",
        content=(NormalizedText(text="Write a long answer."),),
    ),)

    response = gemini_provider.generate_multimodal(messages)

    assert response.text == "Partial answer"
    assert response.status == "incomplete"
