# Copyright (c) 2026 Sugar Labs
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.

import base64

import pytest
from pydantic import ValidationError

from app.multimodal import MAX_IMAGE_BYTES, ImagePart


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.mark.parametrize(
    ("media_type", "data"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\nsmall-png"),
        ("image/jpeg", b"\xff\xd8\xffsmall-jpeg"),
    ],
)
def test_image_part_accepts_matching_png_and_jpeg_signatures(
    media_type: str, data: bytes
) -> None:
    part = ImagePart(
        type="image",
        media_type=media_type,
        source={"type": "base64", "data": encoded(data)},
    )

    assert part.source.decoded_bytes() == data


def test_image_part_rejects_mismatched_png_signature() -> None:
    with pytest.raises(ValidationError, match="PNG signature"):
        ImagePart(
            type="image",
            media_type="image/png",
            source={"type": "base64", "data": encoded(b"not-a-png")},
        )


def test_image_part_rejects_mismatched_jpeg_signature() -> None:
    with pytest.raises(ValidationError, match="JPEG signature"):
        ImagePart(
            type="image",
            media_type="image/jpeg",
            source={"type": "base64", "data": encoded(b"not-a-jpeg")},
        )


def test_image_part_rejects_image_larger_than_one_mib() -> None:
    oversized_png = b"\x89PNG\r\n\x1a\n" + (b"x" * MAX_IMAGE_BYTES)

    with pytest.raises(ValidationError, match="must not exceed 1 MiB"):
        ImagePart(
            type="image",
            media_type="image/png",
            source={"type": "base64", "data": encoded(oversized_png)},
        )
