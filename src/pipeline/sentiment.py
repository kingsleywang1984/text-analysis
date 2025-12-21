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
        """
        Aggregate sentence-level sentiment compound scores into cluster-level sentiment label.
        
        Purpose: Convert a list of sentence-level VADER compound scores (ranging from -1.0 to 1.0)
        into a single categorical sentiment label for the entire cluster using majority voting
        with priority rules for strong negative sentiment.
        
        Input Format:
          - compounds: List[float], VADER compound scores from all sentences in the cluster
            Example: [-0.6249, 0.2263, -0.2960, 0.0]
          - config: AppConfig, contains sentiment threshold configuration:
            - sentiment_strong_negative_threshold: float, threshold for strong negative sentiment
            - sentiment_positive_threshold: float, threshold for positive sentiment
            - sentiment_negative_threshold: float, threshold for negative sentiment
        
        Processing Logic:
          1. If compounds list is empty: return "neutral" (default for empty clusters)
          2. Strong negative priority check:
             - If any sentence has compound < sentiment_strong_negative_threshold:
               immediately return "negative" (strong negative sentiment overrides majority voting)
          3. Count positive and negative sentences:
             - Positive count: number of sentences with compound > sentiment_positive_threshold
             - Negative count: number of sentences with compound < sentiment_negative_threshold
             - Sentences falling between thresholds are not counted in either category
          4. Majority voting with minimum threshold:
             - If positive_count > negative_count AND positive_count >= 1: return "positive"
             - If negative_count > positive_count AND negative_count >= 1: return "negative"
             - Otherwise (tie, or all sentences in neutral range): return "neutral"
        
        Output Format:
          - SentimentLabel: Literal["positive", "neutral", "negative"]
            Example: "negative" (when cluster contains strong negative sentiment or majority negative)
        """
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
