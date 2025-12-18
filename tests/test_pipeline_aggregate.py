from src.config import load_config
from src.pipeline.aggregate import Aggregator
from src.pipeline.clustering import ClusterInternal
from src.pipeline.pipeline import SemanticSentence


def test_aggregator_splits_comment_ids_by_source_and_dedupes():
    cfg = load_config()
    agg = Aggregator(cfg)

    sentences = [
        SemanticSentence(sentence_id="s0", comment_id="c1", text="A", source="baseline", compound=0.1),
        SemanticSentence(sentence_id="s1", comment_id="c1", text="B", source="baseline", compound=-0.2),
        SemanticSentence(sentence_id="s2", comment_id="c2", text="C", source="comparison", compound=0.3),
    ]
    clusters = [ClusterInternal(member_indices=[0, 1, 2], comment_ids=["c1", "c2"])]
    titles = ["T1"]

    reports = agg.build_reports(theme="t", sentences=sentences, clusters=clusters, titles=titles)
    assert len(reports) == 1
    r = reports[0]
    assert list(r.baseline_comment_ids) == ["c1"]
    assert list(r.comparison_comment_ids) == ["c2"]
    assert r.title == "T1"


