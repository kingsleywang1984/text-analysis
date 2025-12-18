from src.config import load_config
from src.pipeline.sentiment import cluster_sentiment


def test_cluster_sentiment_strong_negative_wins():
    cfg = load_config()
    compounds = [-0.7, 0.6, 0.1]
    assert cluster_sentiment(compounds, cfg) == "negative"


def test_cluster_sentiment_positive_dominates():
    cfg = load_config()
    compounds = [0.6, 0.4, -0.1]
    assert cluster_sentiment(compounds, cfg) == "positive"


def test_cluster_sentiment_negative_dominates_without_strong_negative():
    cfg = load_config()
    compounds = [-0.3, -0.25, 0.1]
    assert cluster_sentiment(compounds, cfg) == "negative"


def test_cluster_sentiment_neutral_when_mixed_and_weak():
    cfg = load_config()
    compounds = [0.05, -0.05, 0.1, -0.1]
    assert cluster_sentiment(compounds, cfg) == "neutral"


def test_cluster_sentiment_empty_compounds_is_neutral():
    cfg = load_config()
    assert cluster_sentiment([], cfg) == "neutral"
