def reciprocal_rank_fusion(
    *rankings: list[str],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Combine ranked doc-id lists with Reciprocal Rank Fusion."""
    if not rankings:
        return {}

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return scores
