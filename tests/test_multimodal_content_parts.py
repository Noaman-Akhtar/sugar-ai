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

"""Tests for the basic Phase 1 multimodal content parts."""

import base64

import pytest
from pydantic import ValidationError

from app.multimodal import Base64Source, TextPart


def test_text_part_accepts_non_empty_text():
    part = TextPart.model_validate({"type": "text", "text": "Hello"})

    assert part.type == "text"
    assert part.text == "Hello"


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_text_part_rejects_blank_text(text):
    with pytest.raises(ValidationError, match="must not be blank"):
        TextPart.model_validate({"type": "text", "text": text})


def test_base64_source_decodes_valid_data():
    source_data = b"small media payload"
    source = Base64Source.model_validate({
        "type": "base64",
        "data": base64.b64encode(source_data).decode("ascii"),
    })

    assert source.decoded_bytes() == source_data


@pytest.mark.parametrize("data", ["", "not valid base64!"])
def test_base64_source_rejects_invalid_data(data):
    with pytest.raises(ValidationError):
        Base64Source.model_validate({"type": "base64", "data": data})


def test_content_part_types_are_not_interchangeable():
    with pytest.raises(ValidationError):
        TextPart.model_validate({"type": "base64", "text": "Hello"})

    with pytest.raises(ValidationError):
        Base64Source.model_validate({"type": "text", "data": "SGVsbG8="})
