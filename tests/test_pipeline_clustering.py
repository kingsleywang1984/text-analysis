import json
from pathlib import Path

from src.config import load_config
from src.models.schemas import AnalyzeRequest
from src.pipeline.clustering import SentenceItem, cluster_sentences_greedy_threshold

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> dict:
    p = DATA_DIR / filename
    assert p.exists(), f"Missing example file: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_clustering_produces_non_empty_clusters_from_task_example():
    req = AnalyzeRequest.model_validate(_load_json("input_example.json"))
    cfg = load_config()

    items = [SentenceItem(text=x.sentence, comment_id=x.id, source="baseline") for x in req.baseline[:20]]
    clusters = cluster_sentences_greedy_threshold(items, cfg)

    assert len(clusters) >= 1
    # each cluster must have at least 1 member and 1 comment id
    assert all(len(c.member_indices) >= 1 for c in clusters)
    assert all(len(c.comment_ids) >= 1 for c in clusters)


def test_clustering_is_deterministic():
    req = AnalyzeRequest.model_validate(_load_json("input_example.json"))
    cfg = load_config()

    items = [SentenceItem(text=x.sentence, comment_id=x.id, source="baseline") for x in req.baseline[:25]]

    clusters_a = cluster_sentences_greedy_threshold(items, cfg)
    clusters_b = cluster_sentences_greedy_threshold(items, cfg)

    # Compare stable representation
    rep_a = [(c.member_indices, c.comment_ids) for c in clusters_a]
    rep_b = [(c.member_indices, c.comment_ids) for c in clusters_b]
    assert rep_a == rep_b


def test_cluster_comment_ids_are_deduped_and_sorted():
    cfg = load_config()
    items = [
        SentenceItem(text="login fails", comment_id="c1", source="baseline"),
        SentenceItem(text="cannot sign in", comment_id="c1", source="baseline"),
        SentenceItem(text="payment broken", comment_id="c2", source="baseline"),
    ]
    clusters = cluster_sentences_greedy_threshold(items, cfg)
    # Clustering behaviour depends on the configured similarity threshold, but
    # each cluster should have stable, deduped, sorted comment_ids.
    assert len(clusters) >= 1
    for c in clusters:
        assert c.comment_ids == sorted(set(c.comment_ids))

    # Across all clusters, we should cover the input comment IDs.
    all_ids = sorted({cid for c in clusters for cid in c.comment_ids})
    assert all_ids == ["c1", "c2"]
