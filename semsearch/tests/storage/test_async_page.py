import pytest

from semsearch.storage.models import async_init_db, db as async_db
from semsearch.storage.page import async_read_page_meta, async_save_page


@pytest.mark.asyncio
async def test_async_save_and_read_page_meta(tmp_path):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        saved = await async_save_page(
            "https://example.com/page",
            "<html><body>Hello</body></html>",
            "2026-06-20T00:00:00Z",
        )

        meta = await async_read_page_meta("https://example.com/page")

    assert meta == saved
    assert meta is not None
    assert meta["url"] == "https://example.com/page"


@pytest.mark.asyncio
async def test_async_read_page_meta_missing(tmp_path):
    await async_init_db(tmp_path / "test.db")
    async with async_db:
        assert await async_read_page_meta("https://example.com/missing") is None
