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
from datetime import date, timedelta

import pytest
from fastapi import HTTPException

from app.config import settings
from app.routes import api


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


PNG_DATA = encoded(b"\x89PNG\r\n\x1a\nsmall-png")


def text_part() -> dict:
    return {"type": "text", "text": "Describe this."}


def image_part() -> dict:
    return {
        "type": "image",
        "media_type": "image/png",
        "source": {"type": "base64", "data": PNG_DATA},
    }


def responses_request(*parts: dict) -> api.ResponsesRequest:
    return api.ResponsesRequest(
        messages=[{"role": "user", "content": list(parts)}]
    )


@pytest.fixture(autouse=True)
def clean_quotas():
    api.user_quotas.clear()
    yield
    api.user_quotas.clear()


def test_text_only_request_costs_one_unit() -> None:
    assert api.request_quota_units(responses_request(text_part())) == 1


def test_one_image_request_costs_three_units() -> None:
    request = responses_request(text_part(), image_part())
    assert api.request_quota_units(request) == 3


def test_two_image_request_costs_five_units() -> None:
    request = responses_request(text_part(), image_part(), image_part())
    assert api.request_quota_units(request) == 5


def test_consume_charges_the_requested_units() -> None:
    assert api.consume_quota_units("key", 3)
    assert api.remaining_quota_units("key") == settings.MAX_DAILY_REQUESTS - 3


def test_consume_refuses_without_partial_charge() -> None:
    api.consume_quota_units("key", settings.MAX_DAILY_REQUESTS - 1)
    assert not api.consume_quota_units("key", 2)
    assert api.remaining_quota_units("key") == 1


def test_legacy_check_quota_still_charges_one_unit() -> None:
    assert api.check_quota("key")
    assert api.remaining_quota_units("key") == settings.MAX_DAILY_REQUESTS - 1


def test_legacy_check_quota_still_blocks_at_daily_limit() -> None:
    api.consume_quota_units("key", settings.MAX_DAILY_REQUESTS)
    assert not api.check_quota("key")


def test_quota_resets_on_a_new_day() -> None:
    api.consume_quota_units("key", settings.MAX_DAILY_REQUESTS)
    api.user_quotas["key"]["date"] = date.today() - timedelta(days=1)
    assert api.consume_quota_units("key", 1)
    assert api.remaining_quota_units("key") == settings.MAX_DAILY_REQUESTS - 1


def test_authenticate_api_key_does_not_consume_quota(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "API_KEYS", {"key": {"name": "tester"}}, raising=False
    )
    assert api.authenticate_api_key("key") == "key"
    assert api.remaining_quota_units("key") == settings.MAX_DAILY_REQUESTS


def test_authenticate_api_key_rejects_missing_key() -> None:
    with pytest.raises(HTTPException) as error:
        api.authenticate_api_key(None)
    assert error.value.status_code == 401


def test_authenticate_api_key_rejects_unknown_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "API_KEYS", {}, raising=False)
    with pytest.raises(HTTPException) as error:
        api.authenticate_api_key("unknown")
    assert error.value.status_code == 401


def test_verify_api_key_charges_one_unit(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "API_KEYS", {"key": {"name": "tester"}}, raising=False
    )
    user_info = api.verify_api_key("key")
    assert user_info == {"name": "tester"}
    assert api.remaining_quota_units("key") == settings.MAX_DAILY_REQUESTS - 1


def test_verify_api_key_returns_429_when_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "API_KEYS", {"key": {"name": "tester"}}, raising=False
    )
    api.consume_quota_units("key", settings.MAX_DAILY_REQUESTS)
    with pytest.raises(HTTPException) as error:
        api.verify_api_key("key")
    assert error.value.status_code == 429
