import time

import httpx
import pytest

from semsearch.crawl.crawler import _parse_retry_after, get_rate_limit_wait, is_stale

# ------------------------------------------------------------------
# get_rate_limit_wait
# ------------------------------------------------------------------


def test_no_last_fetch_returns_zero():
    assert get_rate_limit_wait(None) == 0.0


def test_recent_fetch_returns_remaining_wait():
    last = time.time() - 0.3  # fetched 0.3s ago
    wait = get_rate_limit_wait(last, delay=1.0)
    assert 0.6 < wait <= 0.75


def test_old_fetch_returns_zero():
    last = time.time() - 10.0
    assert get_rate_limit_wait(last, delay=1.0) == 0.0


def test_custom_delay():
    last = time.time() - 1.0
    wait = get_rate_limit_wait(last, delay=5.0)
    assert 3.9 < wait <= 4.1


# ------------------------------------------------------------------
# is_stale
# ------------------------------------------------------------------


def test_stale_when_old(monkeypatch):
    meta = {"lastFetchedAt": "2000-01-01T00:00:00Z"}
    assert is_stale(meta) is True


def test_not_stale_when_recent():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"lastFetchedAt": now}
    assert is_stale(meta) is False


def test_stale_when_missing_key():
    assert is_stale({}) is True


def test_stale_when_invalid_date():
    assert is_stale({"lastFetchedAt": "not-a-date"}) is True


# ------------------------------------------------------------------
# _parse_retry_after
# ------------------------------------------------------------------


def _make_response(retry_after: str | None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return httpx.Response(429, headers=headers)


def test_retry_after_seconds():
    resp = _make_response("30")
    result = _parse_retry_after(resp)
    assert result == 30.0


def test_retry_after_none_when_missing():
    resp = _make_response(None)
    assert _parse_retry_after(resp) is None


def test_retry_after_http_date():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    resp = _make_response(format_datetime(future))
    result = _parse_retry_after(resp)
    assert result is not None
    assert 55 < result <= 65


def test_retry_after_invalid_returns_none():
    resp = _make_response("not-a-valid-value")
    assert _parse_retry_after(resp) is None
