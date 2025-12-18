from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import AppConfig

SentimentLabel = Literal["positive", "neutral", "negative"]

_analyzer = SentimentIntensityAnalyzer()

@dataclass(frozen=True, slots=True)
class SentimentEngine:
    """
    Encapsulate sentiment operations. Module-level functions remain for compatibility.
    """

    @staticmethod
    def sentence_compound(text: str) -> float:
        return float(_analyzer.polarity_scores(text)["compound"])

    @staticmethod
    def cluster_sentiment(compounds: List[float], config: AppConfig) -> SentimentLabel:
        if not compounds:
            return "neutral"

        if any(c < config.sentiment_strong_negative_threshold for c in compounds):
            return "negative"

        pos = sum(1 for c in compounds if c > config.sentiment_positive_threshold)
        neg = sum(1 for c in compounds if c < config.sentiment_negative_threshold)

        if pos > neg and pos >= 1:
            return "positive"
        if neg > pos and neg >= 1:
            return "negative"
        return "neutral"


def sentence_compound(text: str) -> float:
    return SentimentEngine.sentence_compound(text)


def cluster_sentiment(compounds: List[float], config: AppConfig) -> SentimentLabel:
    return SentimentEngine.cluster_sentiment(compounds, config)
