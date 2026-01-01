from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, cast

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import AppConfig
from src.logging_utils import log_info, log_warning


def _log(event: str, **fields: Any) -> None:
    log_info(event, **fields)


class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...


def _normalize(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows.astype(np.float32, copy=False)
    rows = rows.astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return rows / norms


@dataclass(frozen=True, slots=True)
class EmbeddingFactory:
    config: AppConfig

    def create(self) -> EmbeddingProvider:
        provider = self.config.embedding_provider.lower()
        if provider == "openai":
            return OpenAIEmbeddingProvider(self.config)
        return TfidfEmbeddingProvider(self.config)


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingProvider:
    config: AppConfig

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        log_info("embedding.openai", embedding_provider="openai")
        base_url = (self.config.embedding_api_base_url or "https://api.openai.com").rstrip("/")
        endpoint = f"{base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.embedding_api_key}",
            "Content-Type": "application/json",
        }
        batch_size = max(1, int(self.config.embedding_api_batch_size))
        timeout = float(self.config.embedding_api_timeout_seconds)
        vectors: List[List[float]] = []

        for batch in _chunks(texts, batch_size):
            payload = {"model": self.config.embedding_model, "input": batch}
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if len(data) != len(batch):
                raise RuntimeError("Embedding API returned unexpected data length")
            vectors.extend(item["embedding"] for item in data)

        arr = np.asarray(vectors, dtype=np.float32)
        return _normalize(arr)


@dataclass(frozen=True, slots=True)
class TfidfEmbeddingProvider:
    config: AppConfig

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        log_info("embedding.tfidf", embedding_provider="tfidf")
        vectorizer = TfidfVectorizer(
            max_features=self.config.embedding_tfidf_max_features,
            ngram_range=(self.config.embedding_tfidf_ngram_min, self.config.embedding_tfidf_ngram_max),
        )
        try:
            # sklearn's typing can be incomplete depending on the environment; we only rely on `.toarray()`.
            matrix = cast(Any, vectorizer.fit_transform(texts))
        except ValueError:
            # Typical: empty/stopword-only vocabulary after preprocessing.
            log_warning(
                "embedding.tfidf.empty_vocab",
                text_count=len(texts),
            )
            return np.zeros((len(texts), 0), dtype=np.float32)

        dense = matrix.toarray()
        return _normalize(np.asarray(dense, dtype=np.float32))


def _chunks(items: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
