"""Orpheus rate-limit cooldown: parse the retry-after hint from a 429 and
fail fast on subsequent calls instead of retrying a doomed request.
"""

import time
from unittest.mock import MagicMock

import httpx

from app.orpheus_tts import OrpheusConfig, OrpheusTTS, _parse_retry_after


def test_parse_retry_after_from_groq_style_message():
    body = "...Requested 110. Please try again in 38m0s. Need more tokens?..."
    response = httpx.Response(429, text=body)
    assert _parse_retry_after(response) == 38 * 60


def test_parse_retry_after_minutes_and_seconds():
    response = httpx.Response(429, text="Please try again in 19m36s.")
    assert _parse_retry_after(response) == 19 * 60 + 36


def test_parse_retry_after_prefers_header():
    response = httpx.Response(429, headers={"retry-after": "45"}, text="ignored")
    assert _parse_retry_after(response) == 45.0


def test_parse_retry_after_none_when_absent():
    response = httpx.Response(429, text="no timing info here")
    assert _parse_retry_after(response) is None


def _fake_429(retry_in: str = "2s") -> MagicMock:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    response.text = f"Please try again in {retry_in}."
    return response


def test_cooldown_short_circuits_without_a_network_call():
    tts = OrpheusTTS(OrpheusConfig(api_key="fake-key"))
    tts._client.post = MagicMock(return_value=_fake_429("2s"))

    first = tts.synthesize("hello")
    assert not first.ok
    assert tts._client.post.call_count == 1

    tts._client.post.reset_mock()
    second = tts.synthesize("hello again")

    assert not second.ok
    assert "rate limited" in second.error
    assert tts._client.post.call_count == 0


def test_cooldown_expires_and_retries():
    tts = OrpheusTTS(OrpheusConfig(api_key="fake-key"))
    tts._client.post = MagicMock(return_value=_fake_429("0s"))

    tts.synthesize("hello")
    # a near-zero cooldown should already have elapsed
    time.sleep(0.05)

    tts._client.post.reset_mock()
    tts.synthesize("hello again")

    assert tts._client.post.call_count == 1
