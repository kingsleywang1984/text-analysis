from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.models.schemas import SentimentLabel

Source = Literal["baseline", "comparison"]


@dataclass(frozen=True, slots=True)
class RawSentence:
    """
    Internal, sentence-level representation.

    - `sentence_id` is deterministic within a single request (stable for debugging/tests).
    - `comment_id` is the input-provided id (comment id).
    """

    sentence_id: str
    comment_id: str
    text: str
    source: Source


@dataclass(frozen=True, slots=True)
class SemanticSentence(RawSentence):
    """
    Sentence-level derived features.
    """

    compound: float  # VADER compound score in [-1, 1]


@dataclass(frozen=True, slots=True)
class ClusterReport:
    """
    Internal reporting view used to format outputs and to drive per-cluster comparison LLM calls.
    """

    title: str
    sentiment: SentimentLabel
    baseline_comment_ids: tuple[str, ...]
    comparison_comment_ids: tuple[str, ...]
    baseline_representative_texts: tuple[str, ...]
    comparison_representative_texts: tuple[str, ...]


def make_sentence_ids(n: int, *, prefix: str = "s") -> list[str]:
    """
    Deterministic IDs for internal tracing.
    """
    return [f"{prefix}{i}" for i in range(n)]


def stable_dedupe_sorted(values: Sequence[str]) -> list[str]:
    return sorted(set(values))

