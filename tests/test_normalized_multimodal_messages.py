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

from app.multimodal import (
    NormalizedImage,
    NormalizedText,
    ResponsesRequest,
    normalize_messages,
)


def test_normalize_messages_preserves_order_and_decodes_images() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\npreview"
    request = ResponsesRequest.model_validate({
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be helpful."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image",
                        "label": "entry-preview",
                        "media_type": "image/png",
                        "source": {
                            "type": "base64",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": "Use simple words."},
                ],
            },
        ],
    })

    messages = normalize_messages(request)

    assert messages[0].role == "system"
    assert messages[0].content == (NormalizedText(text="Be helpful."),)
    assert messages[1].role == "user"
    assert messages[1].content == (
        NormalizedText(text="Describe this."),
        NormalizedImage(
            data=image_bytes,
            media_type="image/png",
            label="entry-preview",
        ),
        NormalizedText(text="Use simple words."),
    )


def test_normalized_image_contains_no_public_base64_source() -> None:
    request = ResponsesRequest.model_validate({
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image",
                "media_type": "image/jpeg",
                "source": {"type": "base64", "data": "/9j/cGhvdG8="},
            }],
        }],
    })

    image = normalize_messages(request)[0].content[0]

    assert isinstance(image, NormalizedImage)
    assert image.data == b"\xff\xd8\xffphoto"
    assert not hasattr(image, "source")
