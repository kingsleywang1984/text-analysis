from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ----------------------------
# Input schema
# ----------------------------

class InputSentence(BaseModel):
    """
    One sentence derived from a comment.
    `id` is the comment identifier (may repeat across multiple sentences).
    """
    model_config = ConfigDict(extra="forbid")

    sentence: str = Field(min_length=1)
    id: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    """
    Task request payload.

    - baseline: required list of sentences
    - comparison: optional list of sentences
    """
    model_config = ConfigDict(extra="forbid")

    surveyTitle: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    baseline: List[InputSentence] = Field(min_length=1)
    comparison: Optional[List[InputSentence]] = None
    query: Optional[str] = None

    @field_validator("comparison", mode="before")
    @classmethod
    def coerce_empty_comparison(cls, value: Optional[List[InputSentence]]) -> Optional[List[InputSentence]]:
        if value == []:
            return None
        return value


# ----------------------------
# Output schema
# ----------------------------

SentimentLabel = Literal["positive", "neutral", "negative"]


class StandaloneCluster(BaseModel):
    """
    Standalone output cluster contract (strict):
    - Must contain ONLY: title, sentiment, keyInsights
    """
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    sentiment: SentimentLabel
    keyInsights: List[str] = Field(min_length=2, max_length=3)


class ComparisonCluster(BaseModel):
    """
    Comparison output cluster contract (strict):
    - Must contain ONLY: title, sentiment, baselineSentences, comparisonSentences, keySimilarities, keyDifferences
    - Must NOT contain keyInsights
    """
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    sentiment: SentimentLabel
    baselineSentences: List[str] = Field(min_length=1)  # comment IDs from baseline cohort (deduped)
    comparisonSentences: List[str] = Field(min_length=1)  # comment IDs from comparison cohort (deduped)
    keySimilarities: List[str] = Field(min_length=1)
    keyDifferences: List[str] = Field(min_length=1)


class AnalyzeResponseStandalone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clusters: List[StandaloneCluster] = Field(min_length=1)


class AnalyzeResponseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clusters: List[ComparisonCluster] = Field(min_length=1)


# ----------------------------
# LLM schema
# ----------------------------

class ClusterLabeling(BaseModel):
    title: str = Field(min_length=3, max_length=80)
    key_insights: List[str]

    @field_validator("key_insights")
    @classmethod
    def _strip_insights(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if s and s.strip()]


class ComparisonSummary(BaseModel):
    key_similarities: List[str]
    key_differences: List[str]

    @field_validator("key_similarities", "key_differences")
    @classmethod
    def _strip_items(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if s and s.strip()]