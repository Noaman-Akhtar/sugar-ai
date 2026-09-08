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

from app.ai import RAGAgent
from app.multimodal import NormalizedMessage, NormalizedText
from app.providers.base import GenerationParams, ProviderResponse


class RecordingProvider:
    def __init__(self) -> None:
        self.received_messages = None
        self.received_params = None
        self.received_response_format = None

    def get_model_name(self) -> str:
        return "recording-model"

    def generate_multimodal(
        self,
        messages: tuple[NormalizedMessage, ...],
        params: GenerationParams | None = None,
        response_format: str = "text",
    ) -> ProviderResponse:
        self.received_messages = messages
        self.received_params = params
        self.received_response_format = response_format
        return ProviderResponse(text="Provider result", status="completed")


def test_rag_agent_delegates_normalized_messages_without_modification() -> None:
    provider = RecordingProvider()
    agent = RAGAgent(provider)
    messages = (NormalizedMessage(
        role="user",
        content=(NormalizedText(text="Describe this image."),),
    ),)
    params = GenerationParams(max_new_tokens=2048, temperature=0.3)

    response = agent.run_multimodal(
        messages,
        params=params,
        response_format="json_object",
    )

    assert provider.received_messages is messages
    assert provider.received_params is params
    assert provider.received_response_format == "json_object"
    assert response == ProviderResponse(text="Provider result", status="completed")
