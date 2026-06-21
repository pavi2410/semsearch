import time

import pytest

from semsearch.crawl.blocks import _5XX_THRESHOLD, BlockList
from semsearch.storage.models import Block, init_db


@pytest.fixture
def blocks(tmp_path):
    init_db(tmp_path / "test.db")
    yield BlockList()
    Block.delete().execute()


# ------------------------------------------------------------------
# is_blocked — clean slate
# ------------------------------------------------------------------


def test_not_blocked_initially(blocks):
    blocked, _ = blocks.is_blocked("https://example.com/page")
    assert not blocked


# ------------------------------------------------------------------
# 403 / 451 — permanent domain block
# ------------------------------------------------------------------


@pytest.mark.parametrize("code", [403, 451])
def test_permanent_domain_block(blocks, code):
    blocks.record("https://example.com/page", code, None)
    blocked, reason = blocks.is_blocked("https://example.com/other")
    assert blocked
    assert str(code) in reason


@pytest.mark.parametrize("code", [403, 451])
def test_permanent_domain_block_persists(tmp_path, code):
    init_db(tmp_path / "test.db")
    b1 = BlockList()
    b1.record("https://example.com/page", code, None)

    b2 = BlockList()
    blocked, _ = b2.is_blocked("https://example.com/anything")
    assert blocked


# ------------------------------------------------------------------
# 429 — temporary domain block
# ------------------------------------------------------------------


def test_429_blocks_domain_temporarily(blocks):
    blocks.record("https://example.com/page", 429, 30.0)
    blocked, reason = blocks.is_blocked("https://example.com/other")
    assert blocked
    assert "429" in reason


def test_429_with_retry_after(blocks):
    blocks.record("https://example.com/page", 429, 30.0)
    entry = Block.get(Block.key == "example.com")
    assert not entry.permanent
    assert entry.until is not None
    assert entry.until > time.time()


def test_429_expires(blocks, monkeypatch):
    blocks.record("https://example.com/page", 429, 1.0)
    monkeypatch.setattr(
        "semsearch.crawl.blocks.time",
        type("t", (), {"time": staticmethod(lambda: time.time() + 10)})(),
    )
    blocked, _ = blocks.is_blocked("https://example.com/other")
    assert not blocked


def test_429_fallback_delay(blocks):
    blocks.record("https://example.com/page", 429, None)
    entry = Block.get(Block.key == "example.com")
    assert entry.until is not None
    assert entry.until >= time.time() + 55


# ------------------------------------------------------------------
# 404 / 410 — permanent URL block
# ------------------------------------------------------------------


@pytest.mark.parametrize("code", [404, 410])
def test_url_block(blocks, code):
    url = "https://example.com/gone"
    blocks.record(url, code, None)
    blocked, reason = blocks.is_blocked(url)
    assert blocked
    assert reason == "url"


@pytest.mark.parametrize("code", [404, 410])
def test_url_block_does_not_block_domain(blocks, code):
    blocks.record("https://example.com/gone", code, None)
    blocked, _ = blocks.is_blocked("https://example.com/other")
    assert not blocked


# ------------------------------------------------------------------
# 5xx — threshold blocking
# ------------------------------------------------------------------


def test_5xx_below_threshold_not_blocked(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD - 1):
        blocks.record(url, 500, None)
    blocked, _ = blocks.is_blocked(url)
    assert not blocked


def test_5xx_at_threshold_blocks_url(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD):
        blocks.record(url, 500, None)
    blocked, reason = blocks.is_blocked(url)
    assert blocked
    assert reason == "url"


def test_5xx_does_not_block_domain(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD):
        blocks.record(url, 500, None)
    blocked, _ = blocks.is_blocked("https://example.com/other")
    assert not blocked


def test_5xx_counts_are_per_url(blocks):
    for _ in range(_5XX_THRESHOLD):
        blocks.record("https://example.com/a", 500, None)
    blocked, _ = blocks.is_blocked("https://example.com/b")
    assert not blocked


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


def test_url_blocks_persist(tmp_path):
    init_db(tmp_path / "test.db")
    b1 = BlockList()
    b1.record("https://example.com/gone", 404, None)

    b2 = BlockList()
    blocked, _ = b2.is_blocked("https://example.com/gone")
    assert blocked


def test_5xx_counts_do_not_persist(tmp_path):
    init_db(tmp_path / "test.db")
    b1 = BlockList()
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD - 1):
        b1.record(url, 500, None)

    b2 = BlockList()
    blocked, _ = b2.is_blocked(url)
    assert not blocked
