def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> dict[str, float]:
    """Combine ranked doc-id lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
