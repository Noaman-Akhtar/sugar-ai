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
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings


MAX_IMAGE_BYTES = 1024 * 1024
MAX_IMAGES_PER_REQUEST = 4
MAX_TOTAL_IMAGE_BYTES = 2 * 1024 * 1024
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


ContentPart: TypeAlias = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ResponseMessage(BaseModel):
    """One role-labelled message in a multimodal response request."""

    role: Literal["system", "user", "assistant"]
    content: list[ContentPart] = Field(min_length=1)

    @model_validator(mode="after")
    def images_must_be_in_user_messages(self) -> "ResponseMessage":
        if self.role != "user" and any(
            isinstance(part, ImagePart) for part in self.content
        ):
            raise ValueError("image parts are allowed only in user messages")
        return self


class GenerationOptions(BaseModel):
    """Provider-neutral controls for one response generation.

    Defaults match GenerationParams so an omitted field behaves the same
    as on the legacy endpoints. Providers that cannot honor a sampling
    control ignore it rather than failing.
    """

    max_new_tokens: int = Field(default=1024, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, gt=0.0, le=2.0)
    truncation: bool = True

    @model_validator(mode="after")
    def max_new_tokens_must_be_within_server_limit(self) -> "GenerationOptions":
        if self.max_new_tokens > settings.MAX_RESPONSE_OUTPUT_TOKENS:
            raise ValueError(
                "max_new_tokens must not exceed the server output token limit"
            )
        return self


class ResponsesRequest(BaseModel):
    """The initial request envelope for the versioned responses endpoint."""

    messages: list[ResponseMessage] = Field(min_length=1)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    response_format: Literal["text", "json_object"] = "text"
    retrieval: bool = False

    @model_validator(mode="after")
    def validate_request_image_limits(self) -> "ResponsesRequest":
        images = [
            part
            for message in self.messages
            for part in message.content
            if isinstance(part, ImagePart)
        ]

        if len(images) > MAX_IMAGES_PER_REQUEST:
            raise ValueError("a request must not contain more than four images")

        total_image_bytes = sum(len(image.source.decoded_bytes()) for image in images)
        if total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("total image data must not exceed 2 MiB")

        return self


@dataclass(frozen=True)
class NormalizedText:
    """Text prepared for a provider without public request metadata."""

    text: str


@dataclass(frozen=True)
class NormalizedImage:
    """Decoded image data prepared for a provider adapter."""

    data: bytes
    media_type: Literal["image/png", "image/jpeg"]
    label: str | None


NormalizedContentPart: TypeAlias = NormalizedText | NormalizedImage


@dataclass(frozen=True)
class NormalizedMessage:
    """A provider-neutral message containing text and decoded images."""

    role: Literal["system", "user", "assistant"]
    content: tuple[NormalizedContentPart, ...]


def normalize_messages(request: ResponsesRequest) -> tuple[NormalizedMessage, ...]:
    """Convert a validated public request into provider-neutral message data."""
    messages = []
    for message in request.messages:
        content: list[NormalizedContentPart] = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append(NormalizedText(text=part.text))
            else:
                content.append(NormalizedImage(
                    data=part.source.decoded_bytes(),
                    media_type=part.media_type,
                    label=part.label,
                ))
        messages.append(NormalizedMessage(role=message.role, content=tuple(content)))
    return tuple(messages)
