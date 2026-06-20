from semsearch.search.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_docs_in_both_rankings():
    bm25 = ["a", "b", "c"]
    semantic = ["b", "a", "d"]
    scores = reciprocal_rank_fusion(bm25, semantic)

    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]
    assert scores["d"] > 0
