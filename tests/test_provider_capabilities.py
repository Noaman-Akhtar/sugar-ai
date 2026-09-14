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

from app.multimodal import NormalizedImage, NormalizedMessage, NormalizedText
from app.providers.base import (
    BaseProvider,
    ProviderResponse,
    UnsupportedModalityError,
)


def provider() -> BaseProvider:
    return BaseProvider(model_name="test-model", api_key="test-key")


def test_base_provider_declares_text_only_input_support() -> None:
    test_provider = provider()

    assert test_provider.supported_input_modalities() == frozenset({"text"})
    assert test_provider.supports_input_modalities({"text"})
    assert not test_provider.supports_input_modalities({"text", "image"})


def test_base_provider_declares_text_only_response_support() -> None:
    test_provider = provider()

    assert test_provider.supports_response_format("text")
    assert not test_provider.supports_response_format("json_object")


def test_base_provider_serves_text_messages_through_chat() -> None:
    test_provider = provider()
    received = {}

    def fake_chat(messages, params=None):
        received["messages"] = messages
        return "Chat answer"

    test_provider.chat = fake_chat
    messages = (NormalizedMessage(
        role="user",
        content=(NormalizedText(text="Hello"),),
    ),)

    result = test_provider.generate_multimodal(messages)

    assert result == ProviderResponse(text="Chat answer", status="completed")
    assert received["messages"] == [{"role": "user", "content": "Hello"}]


def test_base_provider_rejects_images_before_attempting_generation() -> None:
    test_provider = provider()
    messages = (NormalizedMessage(
        role="user",
        content=(NormalizedImage(
            data=b"\x89PNG\r\n\x1a\nimage",
            media_type="image/png",
            label=None,
        ),),
    ),)

    with pytest.raises(UnsupportedModalityError, match="does not support image input"):
        test_provider.generate_multimodal(messages)


def test_provider_response_preserves_incomplete_status() -> None:
    response = ProviderResponse(text="Partial answer", status="incomplete")

    assert response.text == "Partial answer"
    assert response.status == "incomplete"
