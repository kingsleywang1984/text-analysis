from src.config import load_config
from src.pipeline.analyze import RequestAnalyzer
from src.pipeline.clustering import ClusterInternal


def test_overflow_other_returns_max_clusters_with_other_merged():
    cfg = load_config()

    # construct 6 clusters with varying comment_id counts
    clusters = [
        ClusterInternal(member_indices=[0, 1], comment_ids=["a", "b", "c"]),  # 3
        ClusterInternal(member_indices=[2], comment_ids=["d", "e"]),          # 2
        ClusterInternal(member_indices=[3], comment_ids=["f"]),               # 1
        ClusterInternal(member_indices=[4], comment_ids=["g"]),               # 1
        ClusterInternal(member_indices=[5], comment_ids=["h"]),               # 1
        ClusterInternal(member_indices=[6], comment_ids=["i"]),               # 1
    ]

    # assume max_clusters is 3 in .env.test for this test to be meaningful
    # if your .env.test uses 20, set it to 3 for this test, or we can harden it later.
    analyzer = RequestAnalyzer(config=cfg, llm=None)
    out = analyzer._select_top_clusters_with_other(clusters)

    assert len(out) == cfg.cluster_max_clusters
    # last is Other merged tail (must contain union of remaining comment_ids)
    other = out[-1]
    assert set(other.comment_ids).issuperset({"f", "g", "h", "i"})
