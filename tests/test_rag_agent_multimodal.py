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
from app.multimodal import NormalizedImage, NormalizedMessage, NormalizedText
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


class FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


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


def user_message(*parts) -> NormalizedMessage:
    return NormalizedMessage(role="user", content=tuple(parts))


def test_retrieval_appends_matching_context_as_system_message(monkeypatch) -> None:
    provider = RecordingProvider()
    agent = RAGAgent(provider)
    monkeypatch.setattr(
        agent,
        "get_relevant_document",
        lambda query: (FakeDocument("Sugar activities are built in Python."), 0.9),
    )
    messages = (user_message(NormalizedText(text="How do I make an activity?")),)

    agent.run_multimodal(messages, retrieval=True)

    assert len(provider.received_messages) == 2
    context = provider.received_messages[1]
    assert context.role == "system"
    assert "Sugar activities are built in Python." in context.content[0].text


def test_retrieval_queries_with_last_user_message_text_only(monkeypatch) -> None:
    provider = RecordingProvider()
    agent = RAGAgent(provider)
    queries = []

    def record_query(query):
        queries.append(query)
        return None, 0.0

    monkeypatch.setattr(agent, "get_relevant_document", record_query)
    messages = (
        user_message(NormalizedText(text="Old question")),
        NormalizedMessage(
            role="assistant",
            content=(NormalizedText(text="Old answer"),),
        ),
        user_message(
            NormalizedText(text="New question"),
            NormalizedImage(data=b"\x89PNG", media_type="image/png", label=None),
        ),
    )

    agent.run_multimodal(messages, retrieval=True)

    assert queries == ["New question"]
    assert provider.received_messages is messages


def test_retrieval_without_match_leaves_messages_unchanged(monkeypatch) -> None:
    provider = RecordingProvider()
    agent = RAGAgent(provider)
    monkeypatch.setattr(
        agent, "get_relevant_document", lambda query: (None, 0.0)
    )
    messages = (user_message(NormalizedText(text="Hello")),)

    agent.run_multimodal(messages, retrieval=True)

    assert provider.received_messages is messages


def test_retrieval_disabled_never_queries_the_retriever(monkeypatch) -> None:
    provider = RecordingProvider()
    agent = RAGAgent(provider)

    def fail(query):
        raise AssertionError("retriever must not be queried")

    monkeypatch.setattr(agent, "get_relevant_document", fail)
    messages = (user_message(NormalizedText(text="Hello")),)

    agent.run_multimodal(messages)

    assert provider.received_messages is messages
