import json
from pathlib import Path

from src.config import load_config
from src.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponseComparison,
    AnalyzeResponseStandalone,
)
from src.pipeline.analyze import analyze_request

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> dict:
    p = DATA_DIR / filename
    assert p.exists(), f"Missing example file: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_analyze_standalone_task_example_produces_valid_response():
    cfg = load_config()
    req = AnalyzeRequest.model_validate(_load_json("input_example.json"))

    resp = analyze_request(req, cfg)
    # validate schema type
    assert isinstance(resp, AnalyzeResponseStandalone)
    assert len(resp.clusters) >= 1

    for c in resp.clusters:
        assert c.title
        assert c.sentiment in ("positive", "neutral", "negative")
        # keyInsights must be 2–3
        assert 2 <= len(c.keyInsights) <= 3


def test_analyze_comparison_task_example_produces_valid_response():
    cfg = load_config()
    req = AnalyzeRequest.model_validate(_load_json("input_comparison_example.json"))

    resp = analyze_request(req, cfg)
    assert isinstance(resp, AnalyzeResponseComparison)
    assert len(resp.clusters) >= 1
    for c in resp.clusters[:5]:
        assert c.title
        assert c.sentiment in ("positive", "neutral", "negative")
        assert len(c.baselineSentences) >= 1
        assert len(c.comparisonSentences) >= 1
        assert len(c.keySimilarities) >= 1
        assert len(c.keyDifferences) >= 1
