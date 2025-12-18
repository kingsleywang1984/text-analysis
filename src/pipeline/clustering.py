from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from src.config import AppConfig
from src.pipeline.embedding import EmbeddingFactory


@dataclass(frozen=True)
class SentenceItem:
    text: str
    comment_id: str
    source: str  # "baseline" | "comparison"


@dataclass
class ClusterInternal:
    member_indices: List[int]          # indices into input items
    comment_ids: List[str]             # deduped, stable-sorted


@dataclass(frozen=True, slots=True)
class GreedyThresholdClusterer:
    """
    Deterministic greedy clustering. Module-level functions remain for compatibility.
    """

    config: AppConfig

    @staticmethod
    def _cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
        """
        Embeddings are normalized, so cosine similarity = dot product.
        """
        return np.clip(emb @ emb.T, -1.0, 1.0)

    def cluster(self, items: List[SentenceItem]) -> List[ClusterInternal]:
        """
        Deterministic greedy clustering:
        - Compute embeddings (normalized)
        - Compute cosine similarity matrix
        - While unassigned sentences remain:
            - pick a deterministic seed (highest total similarity to remaining)
            - group all remaining sentences with sim >= threshold
        - Any leftovers become singletons

        Note: This produces non-overlapping sentence clusters. Overlap at the ID level
        can still occur naturally when multiple sentences share the same comment_id and
        end up in different clusters.
        """
        if not items:
            return []

        provider = EmbeddingFactory(self.config).create()
        texts = [it.text for it in items]
        emb = provider.embed(texts)
        sim = self._cosine_sim_matrix(emb)

        n = len(items)
        remaining: Set[int] = set(range(n))
        clusters: List[ClusterInternal] = []

        while remaining and len(clusters) < self.config.cluster_max_clusters:
            remaining_list = sorted(remaining)

            # deterministic seed: maximize total similarity to remaining
            def seed_score(i: int) -> float:
                return float(sim[i, remaining_list].sum())

            seed = max(remaining_list, key=seed_score)

            members = [
                j
                for j in remaining_list
                if float(sim[seed, j]) >= self.config.cluster_similarity_threshold
            ]
            for j in members:
                remaining.discard(j)

            comment_ids = sorted({items[j].comment_id for j in members})
            clusters.append(ClusterInternal(member_indices=members, comment_ids=comment_ids))

        # leftover → singleton clusters (deterministic order)
        for i in sorted(remaining):
            clusters.append(ClusterInternal(member_indices=[i], comment_ids=[items[i].comment_id]))

        # stable ordering for testability: sort clusters by (size desc, first comment_id asc)
        clusters.sort(key=lambda c: (-len(c.member_indices), c.comment_ids[0]))
        return clusters


def cluster_sentences_greedy_threshold(
    items: List[SentenceItem],
    config: AppConfig
) -> List[ClusterInternal]:
    return GreedyThresholdClusterer(config).cluster(items)
