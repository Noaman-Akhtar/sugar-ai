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

"""Provider-neutral content parts for versioned multimodal requests."""

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_IMAGE_BYTES = 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"


class TextPart(BaseModel):
    """A non-empty text item in a multimodal message."""

    type: Literal["text"]
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text content must not be blank")
        return value


class Base64Source(BaseModel):

    type: Literal["base64"]
    data: str

    @field_validator("data")
    @classmethod
    def data_must_be_base64(cls, value: str) -> str:
        if not value:
            raise ValueError("base64 data must not be empty")
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("media data must be valid Base64") from error
        return value

    def decoded_bytes(self) -> bytes:
        """Return decoded bytes without logging the original media data."""
        return base64.b64decode(self.data, validate=True)


class ImagePart(BaseModel):
    """An inline PNG or JPEG image supplied by an activity."""

    type: Literal["image"]
    media_type: Literal["image/png", "image/jpeg"]
    source: Base64Source
    label: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_image(self) -> "ImagePart":
        image_bytes = self.source.decoded_bytes()

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image data must not exceed 1 MiB")

        if self.media_type == "image/png" and not image_bytes.startswith(_PNG_SIGNATURE):
            raise ValueError("image/png data must have a PNG signature")

        if self.media_type == "image/jpeg" and not image_bytes.startswith(_JPEG_SIGNATURE):
            raise ValueError("image/jpeg data must have a JPEG signature")

        return self
