import zstandard as zstd

_COMPRESSOR = zstd.ZstdCompressor(level=3)
_DECOMPRESSOR = zstd.ZstdDecompressor()


def compress_html(html: str) -> bytes:
    return _COMPRESSOR.compress(html.encode())


def decompress_html(body: bytes) -> str:
    return _DECOMPRESSOR.decompress(body).decode()
