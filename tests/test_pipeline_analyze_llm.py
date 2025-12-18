import json
from pathlib import Path

from src.config import load_config
from src.llm.client import FakeLLMClient
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


def test_analyze_uses_llm_first_for_titles_and_insights():
    cfg = load_config()
    req = AnalyzeRequest.model_validate(_load_json("input_example.json"))

    resp = analyze_request(req, cfg, llm=FakeLLMClient())
    assert isinstance(resp, AnalyzeResponseStandalone)
    assert len(resp.clusters) >= 1

    # FakeLLMClient produces titles prefixed with "<theme>: ..."
    # Ensure at least one cluster got LLM title (not fallback "theme cluster X")
    assert any(c.title.startswith(f"{req.theme}:") for c in resp.clusters)

    # Ensure keyInsights satisfy configured budget (2-3) and are non-empty
    for c in resp.clusters[:5]:
        assert 2 <= len(c.keyInsights) <= 3
        assert all(isinstance(x, str) and x.strip() for x in c.keyInsights)


def test_analyze_comparison_uses_llm_per_cluster_for_similarities_and_differences():
    cfg = load_config()
    req = AnalyzeRequest.model_validate(_load_json("input_comparison_example.json"))

    resp = analyze_request(req, cfg, llm=FakeLLMClient())
    assert isinstance(resp, AnalyzeResponseComparison)
    assert len(resp.clusters) >= 1
    # FakeLLMClient produces per-cluster similarities mentioning the cluster title
    assert any("Both cohorts mention" in s for c in resp.clusters for s in c.keySimilarities)
    # And it should also influence the comparison cluster title via label_cluster
    assert any(c.title.startswith(f"{req.theme}:") for c in resp.clusters)


def test_analyze_comparison_falls_back_deterministic_when_llm_missing():
    cfg = load_config()
    req = AnalyzeRequest.model_validate(_load_json("input_comparison_example.json"))

    resp = analyze_request(req, cfg, llm=None)
    assert isinstance(resp, AnalyzeResponseComparison)
    assert len(resp.clusters) >= 1
    # Deterministic fallback includes a standard prefix
    assert any("Both cohorts discuss" in s for c in resp.clusters for s in c.keySimilarities)
