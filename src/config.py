# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    # Embedding
    embedding_model: str
    embedding_provider: str  # "tfidf" | "openai"
    embedding_api_base_url: str | None
    embedding_api_key: str | None
    embedding_api_timeout_seconds: float
    embedding_api_batch_size: int
    embedding_tfidf_max_features: int | None
    embedding_tfidf_ngram_min: int
    embedding_tfidf_ngram_max: int

    # Clustering
    cluster_similarity_threshold: float
    cluster_max_clusters: int
    cluster_overflow_strategy: str  # "OTHER" | "DROP"

    # Sentiment
    sentiment_strong_negative_threshold: float
    sentiment_positive_threshold: float
    sentiment_negative_threshold: float

    # LLM output limits
    cluster_insights_min: int
    cluster_insights_max: int
    comparison_similarities_min: int
    comparison_similarities_max: int
    comparison_differences_min: int
    comparison_differences_max: int

    # LLM (used later)
    llm_provider: str  # "none" | "openai_compatible"
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_timeout_seconds: float
    llm_temperature: float
    llm_max_retries: int

    # LLM bounded controls (LLM-first strategy)
    llm_max_clusters: int
    llm_representative_sentences_per_cluster: int


@dataclass(frozen=True, slots=True)
class EnvConfigLoader:
    """
    Load `AppConfig` from environment variables.

    The module-level `load_config()` function remains the public entrypoint.
    """

    @staticmethod
    def _env(name: str) -> str | None:
        v = os.getenv(name)
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    @classmethod
    def _env_float(cls, name: str) -> float | None:
        v = cls._env(name)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            raise RuntimeError(f"{name} must be a float, got: {v}")

    @classmethod
    def _env_int(cls, name: str) -> int | None:
        v = cls._env(name)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            raise RuntimeError(f"{name} must be an integer, got: {v}")

    def load(self) -> AppConfig:
        embedding_provider = (self._env("EMBEDDING_PROVIDER") or "tfidf").strip().lower()
        if embedding_provider not in ("tfidf", "openai"):
            raise RuntimeError("EMBEDDING_PROVIDER must be one of: tfidf, openai")

        embedding_model = self._env("EMBEDDING_MODEL")
        if not embedding_model:
            raise RuntimeError("EMBEDDING_MODEL must be set (e.g. via .env or runtime environment)")

        embedding_api_base_url = self._env("EMBEDDING_API_BASE_URL") or "https://api.openai.com"
        embedding_api_key = self._env("EMBEDDING_API_KEY")
        embedding_api_timeout = self._env_float("EMBEDDING_API_TIMEOUT_SECONDS") or 30.0
        if embedding_api_timeout <= 0:
            raise RuntimeError("EMBEDDING_API_TIMEOUT_SECONDS must be > 0")

        embedding_api_batch_size = self._env_int("EMBEDDING_API_BATCH_SIZE") or 256
        if embedding_api_batch_size <= 0:
            raise RuntimeError("EMBEDDING_API_BATCH_SIZE must be a positive integer")

        embedding_tfidf_max_features = self._env_int("EMBEDDING_TFIDF_MAX_FEATURES")
        if embedding_tfidf_max_features is not None and embedding_tfidf_max_features <= 0:
            raise RuntimeError("EMBEDDING_TFIDF_MAX_FEATURES must be > 0 when set")

        embedding_tfidf_ngram_min = self._env_int("EMBEDDING_TFIDF_NGRAM_MIN") or 1
        embedding_tfidf_ngram_max = self._env_int("EMBEDDING_TFIDF_NGRAM_MAX") or embedding_tfidf_ngram_min
        if embedding_tfidf_ngram_min <= 0 or embedding_tfidf_ngram_max < embedding_tfidf_ngram_min:
            raise RuntimeError("TF-IDF ngram settings must satisfy: min > 0 and max >= min")

        if embedding_provider == "openai" and not embedding_api_key:
            raise RuntimeError("EMBEDDING_API_KEY must be set when EMBEDDING_PROVIDER=openai")

        similarity_threshold = self._env_float("CLUSTER_SIMILARITY_THRESHOLD")
        if similarity_threshold is None or not (0.0 < similarity_threshold <= 1.0):
            raise RuntimeError("CLUSTER_SIMILARITY_THRESHOLD must be set and in (0.0, 1.0]")

        max_clusters = self._env_int("CLUSTER_MAX_CLUSTERS")
        if max_clusters is None or max_clusters <= 0:
            raise RuntimeError("CLUSTER_MAX_CLUSTERS must be a positive integer")

        overflow = self._env("CLUSTER_OVERFLOW_STRATEGY") or "OTHER"
        overflow = overflow.upper()
        if overflow not in ("OTHER", "DROP"):
            raise RuntimeError("CLUSTER_OVERFLOW_STRATEGY must be OTHER or DROP")

        strong_neg = self._env_float("SENTIMENT_STRONG_NEGATIVE_THRESHOLD")
        if strong_neg is None:
            raise RuntimeError("SENTIMENT_STRONG_NEGATIVE_THRESHOLD must be set")

        pos_th = self._env_float("SENTIMENT_POSITIVE_THRESHOLD")
        neg_th = self._env_float("SENTIMENT_NEGATIVE_THRESHOLD")
        if pos_th is None or neg_th is None:
            raise RuntimeError("SENTIMENT_POSITIVE_THRESHOLD and SENTIMENT_NEGATIVE_THRESHOLD must be set")

        if not (neg_th < 0 < pos_th):
            raise RuntimeError("Sentiment thresholds must satisfy: negative < 0 < positive")

        ins_min = self._env_int("CLUSTER_INSIGHTS_MIN")
        ins_max = self._env_int("CLUSTER_INSIGHTS_MAX")
        if ins_min is None or ins_max is None or ins_min <= 0 or ins_min > ins_max:
            raise RuntimeError("CLUSTER_INSIGHTS_MIN/MAX must be set and valid")

        sim_min = self._env_int("COMPARISON_SIMILARITIES_MIN")
        sim_max = self._env_int("COMPARISON_SIMILARITIES_MAX")
        if sim_min is None or sim_max is None or sim_min <= 0 or sim_min > sim_max:
            raise RuntimeError("COMPARISON_SIMILARITIES_MIN/MAX must be set and valid")

        dif_min = self._env_int("COMPARISON_DIFFERENCES_MIN")
        dif_max = self._env_int("COMPARISON_DIFFERENCES_MAX")
        if dif_min is None or dif_max is None or dif_min <= 0 or dif_min > dif_max:
            raise RuntimeError("COMPARISON_DIFFERENCES_MIN/MAX must be set and valid")

        llm_provider = (self._env("LLM_PROVIDER") or "none").strip().lower()
        if llm_provider not in ("none", "openai_compatible"):
            raise RuntimeError("LLM_PROVIDER must be one of: none, openai_compatible")

        llm_base_url = self._env("LLM_BASE_URL")
        llm_api_key = self._env("LLM_API_KEY")
        llm_model = self._env("LLM_MODEL")

        # Deploy-time validation: if a provider is selected, required config must be present.
        if llm_provider != "none" and not (llm_base_url and llm_api_key and llm_model):
            raise RuntimeError("LLM_BASE_URL / LLM_API_KEY / LLM_MODEL must be set when LLM_PROVIDER is enabled")

        llm_timeout = self._env_float("LLM_TIMEOUT_SECONDS")
        if llm_timeout is None:
            raise RuntimeError("LLM_TIMEOUT_SECONDS must be set")
        if llm_timeout <= 0:
            raise RuntimeError("LLM_TIMEOUT_SECONDS must be > 0")

        llm_temp = self._env_float("LLM_TEMPERATURE")
        if llm_temp is None:
            raise RuntimeError("LLM_TEMPERATURE must be set")
        if not (0.0 <= llm_temp <= 2.0):
            raise RuntimeError("LLM_TEMPERATURE must be between 0.0 and 2.0")

        llm_retries = self._env_int("LLM_MAX_RETRIES")
        if llm_retries is None:
            raise RuntimeError("LLM_MAX_RETRIES must be set")
        if llm_retries < 0:
            raise RuntimeError("LLM_MAX_RETRIES must be >= 0")

        llm_max_clusters = self._env_int("LLM_MAX_CLUSTERS") or 10
        if llm_max_clusters <= 0:
            raise RuntimeError("LLM_MAX_CLUSTERS must be a positive integer")

        llm_rep = self._env_int("LLM_REPRESENTATIVE_SENTENCES_PER_CLUSTER") or 10
        if llm_rep <= 0:
            raise RuntimeError("LLM_REPRESENTATIVE_SENTENCES_PER_CLUSTER must be a positive integer")

        return AppConfig(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_api_base_url=embedding_api_base_url,
            embedding_api_key=embedding_api_key,
            embedding_api_timeout_seconds=embedding_api_timeout,
            embedding_api_batch_size=embedding_api_batch_size,
            embedding_tfidf_max_features=embedding_tfidf_max_features,
            embedding_tfidf_ngram_min=embedding_tfidf_ngram_min,
            embedding_tfidf_ngram_max=embedding_tfidf_ngram_max,
            cluster_similarity_threshold=similarity_threshold,
            cluster_max_clusters=max_clusters,
            cluster_overflow_strategy=overflow,
            sentiment_strong_negative_threshold=strong_neg,
            sentiment_positive_threshold=pos_th,
            sentiment_negative_threshold=neg_th,
            cluster_insights_min=ins_min,
            cluster_insights_max=ins_max,
            comparison_similarities_min=sim_min,
            comparison_similarities_max=sim_max,
            comparison_differences_min=dif_min,
            comparison_differences_max=dif_max,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_timeout_seconds=llm_timeout,
            llm_temperature=llm_temp,
            llm_max_retries=llm_retries,
            llm_max_clusters=llm_max_clusters,
            llm_representative_sentences_per_cluster=llm_rep,
        )


def load_config() -> AppConfig:
    return EnvConfigLoader().load()
