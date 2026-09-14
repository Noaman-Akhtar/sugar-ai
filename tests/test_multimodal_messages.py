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
from pydantic import ValidationError

from app.multimodal import (
    MAX_IMAGES_PER_REQUEST,
    MAX_TOTAL_IMAGE_BYTES,
    ResponseMessage,
    ResponsesRequest,
)


def image_part(data: bytes = b"\x89PNG\r\n\x1a\nimage") -> dict[str, object]:
    return {
        "type": "image",
        "media_type": "image/png",
        "source": {
            "type": "base64",
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def test_response_message_accepts_mixed_user_content() -> None:
    message = ResponseMessage.model_validate({
        "role": "user",
        "content": [{"type": "text", "text": "Describe this"}, image_part()],
    })

    assert len(message.content) == 2


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_response_message_rejects_images_outside_user_role(role: str) -> None:
    with pytest.raises(ValidationError, match="only in user messages"):
        ResponseMessage.model_validate({"role": role, "content": [image_part()]})


def test_response_message_requires_content() -> None:
    with pytest.raises(ValidationError):
        ResponseMessage.model_validate({"role": "user", "content": []})


def test_responses_request_rejects_more_than_four_images() -> None:
    with pytest.raises(ValidationError, match="more than four images"):
        ResponsesRequest.model_validate({
            "messages": [{
                "role": "user",
                "content": [image_part()] * (MAX_IMAGES_PER_REQUEST + 1),
            }],
        })


def test_responses_request_rejects_images_exceeding_total_limit() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + (b"x" * (MAX_TOTAL_IMAGE_BYTES // 3))

    with pytest.raises(ValidationError, match="total image data must not exceed 2 MiB"):
        ResponsesRequest.model_validate({
            "messages": [{
                "role": "user",
                "content": [image_part(image_bytes)] * 3,
            }],
        })
