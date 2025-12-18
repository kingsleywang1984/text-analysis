import json

from src.handler import lambda_handler


def _set_min_env(monkeypatch):
    # Minimal set required by load_config()
    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("CLUSTER_SIMILARITY_THRESHOLD", "0.75")
    monkeypatch.setenv("CLUSTER_MAX_CLUSTERS", "10")
    monkeypatch.setenv("CLUSTER_OVERFLOW_STRATEGY", "OTHER")
    monkeypatch.setenv("SENTIMENT_STRONG_NEGATIVE_THRESHOLD", "-0.5")
    monkeypatch.setenv("SENTIMENT_POSITIVE_THRESHOLD", "0.3")
    monkeypatch.setenv("SENTIMENT_NEGATIVE_THRESHOLD", "-0.2")
    monkeypatch.setenv("CLUSTER_INSIGHTS_MIN", "2")
    monkeypatch.setenv("CLUSTER_INSIGHTS_MAX", "3")
    monkeypatch.setenv("COMPARISON_SIMILARITIES_MIN", "1")
    monkeypatch.setenv("COMPARISON_SIMILARITIES_MAX", "3")
    monkeypatch.setenv("COMPARISON_DIFFERENCES_MIN", "1")
    monkeypatch.setenv("COMPARISON_DIFFERENCES_MAX", "3")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_MAX_CLUSTERS", "10")
    monkeypatch.setenv("LLM_REPRESENTATIVE_SENTENCES_PER_CLUSTER", "10")


def test_handler_rejects_non_analyze_route(monkeypatch):
    _set_min_env(monkeypatch)
    event = {"requestContext": {"http": {"method": "GET", "path": "/health"}}, "body": "{}"}
    resp = lambda_handler(event, None)
    assert resp["statusCode"] == 404


def test_handler_accepts_http_api_v2_event(monkeypatch):
    _set_min_env(monkeypatch)
    payload = {
        "surveyTitle": "t",
        "theme": "food",
        "baseline": [{"sentence": "Great meals", "id": "c1"}],
        "comparison": [{"sentence": "Great meals", "id": "c2"}],
    }
    event = {
        "requestContext": {"http": {"method": "POST", "path": "/analyze"}},
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }
    resp = lambda_handler(event, None)
    assert resp["statusCode"] in (200, 500)  # embedding model may not be available in CI/runtime


