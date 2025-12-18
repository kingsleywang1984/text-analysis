import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponseComparison,
    AnalyzeResponseStandalone,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> dict:
    p = DATA_DIR / filename
    assert p.exists(), f"Missing example file: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_request_parses_task_input_example():
    payload = _load_json("input_example.json")
    req = AnalyzeRequest.model_validate(payload)

    assert req.surveyTitle
    assert req.theme
    assert len(req.baseline) > 0
    assert req.comparison is None


def test_request_parses_task_input_comparison_example():
    payload = _load_json("input_comparison_example.json")
    req = AnalyzeRequest.model_validate(payload)

    assert req.surveyTitle
    assert req.theme
    assert len(req.baseline) > 0
    assert req.comparison is not None
    assert len(req.comparison) > 0


def test_request_rejects_extra_fields_based_on_task_example():
    payload = _load_json("input_example.json")
    payload["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_standalone_response_schema_accepts_minimal_valid_shape():
    # Construct a minimal response based on the task's expected output shape.
    # (We cannot validate against their exact output values without running the pipeline.)
    resp = AnalyzeResponseStandalone.model_validate({
        "clusters": [
            {
                "title": "Example cluster title",
                "sentiment": "neutral",
                "keyInsights": ["Insight 1", "Insight 2"],
            }
        ]
    })
    assert resp.clusters[0].sentiment in ("positive", "neutral", "negative")


def test_comparison_response_schema_accepts_minimal_valid_shape():
    resp = AnalyzeResponseComparison.model_validate({
        "clusters": [
            {
                "title": "Example cluster title",
                "sentiment": "negative",
                "baselineSentences": ["comment-1"],
                "comparisonSentences": ["comment-2"],
                "keySimilarities": ["Similarity 1"],
                "keyDifferences": ["Difference 1"],
            }
        ],
    })
    assert len(resp.clusters) >= 1


def test_cluster_keyinsights_must_be_2_to_3():
    # Variant derived from expected output shape: keyInsights too short
    with pytest.raises(ValidationError):
        AnalyzeResponseStandalone.model_validate({
            "clusters": [
                {
                    "title": "Bad cluster",
                    "sentiment": "negative",
                    "keyInsights": ["Only one"],  # invalid
                }
            ]
        })

    # Variant: keyInsights too long
    with pytest.raises(ValidationError):
        AnalyzeResponseStandalone.model_validate({
            "clusters": [
                {
                    "title": "Bad cluster",
                    "sentiment": "negative",
                    "keyInsights": ["1", "2", "3", "4"],  # invalid
                }
            ]
        })


def test_comparison_cluster_must_reject_keyinsights():
    with pytest.raises(ValidationError):
        AnalyzeResponseComparison.model_validate({
            "clusters": [
                {
                    "title": "Bad comparison cluster",
                    "sentiment": "neutral",
                    "baselineSentences": ["comment-1"],
                    "comparisonSentences": ["comment-2"],
                    "keyInsights": ["should not be here"],
                    "keySimilarities": ["Similarity 1"],
                    "keyDifferences": ["Difference 1"],
                }
            ]
        })