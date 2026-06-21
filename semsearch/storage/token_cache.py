from .models import TokenCache


def load_tokens(content_hash: str) -> list[str] | None:
    try:
        row = (
            TokenCache.select(TokenCache.tokens.json())
            .where(TokenCache.content_hash == content_hash)
            .get()
        )
    except TokenCache.DoesNotExist:
        return None
    return row.tokens


def save_tokens(content_hash: str, tokens: list[str]) -> None:
    TokenCache.replace(content_hash=content_hash, tokens=tokens).execute()
