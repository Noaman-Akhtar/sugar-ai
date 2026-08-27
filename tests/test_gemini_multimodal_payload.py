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

from app.multimodal import NormalizedImage, NormalizedMessage, NormalizedText
from app.providers.gemini import GeminiProvider


def provider() -> GeminiProvider:
    return GeminiProvider(model_name="test-model", api_key="test-key")


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
