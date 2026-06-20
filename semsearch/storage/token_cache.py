import json

from .models import SyncTokenCache as TokenCache


def load_tokens(content_hash: str) -> list[str] | None:
    try:
        row = TokenCache.get_by_id(content_hash)
    except TokenCache.DoesNotExist:
        return None
    return json.loads(row.tokens)


def save_tokens(content_hash: str, tokens: list[str]) -> None:
    TokenCache.replace(content_hash=content_hash, tokens=json.dumps(tokens)).execute()
