from semsearch.search.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_docs_in_both_rankings():
    bm25 = ["a", "b", "c"]
    semantic = ["b", "a", "d"]
    scores = reciprocal_rank_fusion(bm25, semantic)

    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]
    assert scores["d"] > 0


def test_reciprocal_rank_fusion_supports_weighted_semantic_ranking():
    bm25 = ["a", "b", "c"]
    semantic = ["c", "a", "b"]
    balanced = reciprocal_rank_fusion(bm25, semantic, weights=[1.0, 1.0])
    semantic_heavy = reciprocal_rank_fusion(bm25, semantic, weights=[1.0, 2.0])

    assert semantic_heavy["c"] > balanced["c"]
    assert semantic_heavy["a"] > balanced["a"]
