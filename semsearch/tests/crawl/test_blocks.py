import time

import pytest

from semsearch.crawl.blocks import _5XX_THRESHOLD, BlockList
from semsearch.storage.models import Block, async_init_db, db as async_db


@pytest.fixture
async def initialized_db(tmp_path):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        yield
        await async_db.aexecute(Block.delete())


@pytest.fixture
def blocks(initialized_db):
    return BlockList()


# ------------------------------------------------------------------
# is_blocked — clean slate
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_blocked_initially(blocks):
    blocked, _ = await blocks.is_blocked("https://example.com/page")
    assert not blocked


# ------------------------------------------------------------------
# 403 / 451 — permanent domain block
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [403, 451])
async def test_permanent_domain_block(blocks, code):
    await blocks.record("https://example.com/page", code, None)
    blocked, reason = await blocks.is_blocked("https://example.com/other")
    assert blocked
    assert str(code) in reason


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [403, 451])
async def test_permanent_domain_block_persists(tmp_path, code):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        b1 = BlockList()
        await b1.record("https://example.com/page", code, None)

        b2 = BlockList()
        blocked, _ = await b2.is_blocked("https://example.com/anything")
        assert blocked


# ------------------------------------------------------------------
# 429 — temporary domain block
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_429_blocks_domain_temporarily(blocks):
    await blocks.record("https://example.com/page", 429, 30.0)
    blocked, reason = await blocks.is_blocked("https://example.com/other")
    assert blocked
    assert "429" in reason


@pytest.mark.asyncio
async def test_429_with_retry_after(blocks):
    await blocks.record("https://example.com/page", 429, 30.0)
    entry = await async_db.get(Block.select().where(Block.key == "example.com"))
    assert not entry.permanent
    assert entry.until is not None
    assert entry.until > time.time()


@pytest.mark.asyncio
async def test_429_expires(blocks, monkeypatch):
    await blocks.record("https://example.com/page", 429, 1.0)
    monkeypatch.setattr(
        "semsearch.crawl.blocks.time",
        type("t", (), {"time": staticmethod(lambda: time.time() + 10)})(),
    )
    blocked, _ = await blocks.is_blocked("https://example.com/other")
    assert not blocked


@pytest.mark.asyncio
async def test_429_fallback_delay(blocks):
    await blocks.record("https://example.com/page", 429, None)
    entry = await async_db.get(Block.select().where(Block.key == "example.com"))
    assert entry.until is not None
    assert entry.until >= time.time() + 55


# ------------------------------------------------------------------
# 404 / 410 — permanent URL block
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [404, 410])
async def test_url_block(blocks, code):
    url = "https://example.com/gone"
    await blocks.record(url, code, None)
    blocked, reason = await blocks.is_blocked(url)
    assert blocked
    assert reason == "url"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [404, 410])
async def test_url_block_does_not_block_domain(blocks, code):
    await blocks.record("https://example.com/gone", code, None)
    blocked, _ = await blocks.is_blocked("https://example.com/other")
    assert not blocked


# ------------------------------------------------------------------
# 5xx — threshold blocking
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5xx_below_threshold_not_blocked(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD - 1):
        await blocks.record(url, 500, None)
    blocked, _ = await blocks.is_blocked(url)
    assert not blocked


@pytest.mark.asyncio
async def test_5xx_at_threshold_blocks_url(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD):
        await blocks.record(url, 500, None)
    blocked, reason = await blocks.is_blocked(url)
    assert blocked
    assert reason == "url"


@pytest.mark.asyncio
async def test_5xx_does_not_block_domain(blocks):
    url = "https://example.com/flaky"
    for _ in range(_5XX_THRESHOLD):
        await blocks.record(url, 500, None)
    blocked, _ = await blocks.is_blocked("https://example.com/other")
    assert not blocked


@pytest.mark.asyncio
async def test_5xx_counts_are_per_url(blocks):
    for _ in range(_5XX_THRESHOLD):
        await blocks.record("https://example.com/a", 500, None)
    blocked, _ = await blocks.is_blocked("https://example.com/b")
    assert not blocked


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_blocks_persist(tmp_path):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        b1 = BlockList()
        await b1.record("https://example.com/gone", 404, None)

        b2 = BlockList()
        blocked, _ = await b2.is_blocked("https://example.com/gone")
        assert blocked


@pytest.mark.asyncio
async def test_5xx_counts_do_not_persist(tmp_path):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        b1 = BlockList()
        url = "https://example.com/flaky"
        for _ in range(_5XX_THRESHOLD - 1):
            await b1.record(url, 500, None)

        b2 = BlockList()
        blocked, _ = await b2.is_blocked(url)
        assert not blocked
