from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from src.config import AppConfig
from src.logging_utils import log_info
from src.pipeline.embedding import EmbeddingFactory
from src.pipeline.pipeline import Source


@dataclass(frozen=True)
class SentenceItem:
    text: str
    comment_id: str
    source: Source  # "baseline" | "comparison"


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
        Deterministic greedy clustering algorithm for grouping semantically similar sentences.
        
        Purpose: Group sentences into clusters based on semantic similarity using a greedy
        threshold-based approach. The algorithm ensures deterministic results by using stable
        ordering and selection criteria.
        
        Note: This produces non-overlapping sentence clusters. Overlap at the ID level
        can still occur naturally when multiple sentences share the same comment_id and
        end up in different clusters.
        
        Input Format:
          - items: List[SentenceItem], sentences to cluster
            Each contains: {text: str, comment_id: str, source: str}
            Example: [SentenceItem(text="Withholding my money", comment_id="uuid-123", source="baseline"), ...]
          - self.config: AppConfig, contains clustering configuration:
            - cluster_similarity_threshold: float, minimum similarity (0.0-1.0) to group sentences
            - cluster_max_clusters: int, maximum number of clusters to create
        
        Output Format:
          - List[ClusterInternal], non-overlapping sentence clusters
            Each contains:
              - member_indices: List[int], sentence indices in the cluster (sorted)
              - comment_ids: List[str], deduplicated comment IDs from cluster members (sorted)
            Example: [
              ClusterInternal(member_indices=[0, 1, 2], comment_ids=["uuid-123", "uuid-456"]),
              ClusterInternal(member_indices=[3], comment_ids=["uuid-789"]),
              ...
            ]
        """
        # ========================================================================
        # STEP 1: Input Validation
        # ========================================================================
        # Purpose: Handle edge case of empty input list
        #
        # Processing Logic:
        #   - If items list is empty: return empty list immediately
        #   - Prevents unnecessary processing and ensures consistent behavior
        if not items:
            return []

        # ========================================================================
        # STEP 2: Embedding Generation
        # ========================================================================
        # Purpose: Convert text into numerical vectors for similarity computation
        #
        # Processing Logic:
        #   1. Create embedding provider using EmbeddingFactory
        #      - Provider type depends on config (TF-IDF or OpenAI embedding)
        #   2. Extract text content from all SentenceItem objects
        #      - Collect only the text field, ignoring comment_id and source
        #   3. Generate normalized embedding vectors for all sentences
        #      - Each sentence is converted to a fixed-size numerical vector
        #      - Embeddings are normalized (unit length) for efficient cosine similarity computation
        provider = EmbeddingFactory(self.config).create()
        texts = [it.text for it in items]
        emb = provider.embed(texts)

        # ========================================================================
        # STEP 3: Similarity Matrix Computation
        # ========================================================================
        # Purpose: Pre-compute all pairwise similarities for efficient cluster formation
        #
        # Processing Logic:
        #   1. Compute cosine similarity matrix between all sentence pairs
        #      - Since embeddings are normalized, cosine similarity = dot product (emb @ emb.T)
        #   2. Clip values to [-1.0, 1.0] range for numerical stability
        #      - Prevents floating-point errors from exceeding valid similarity range
        #   3. Result is a symmetric matrix where sim[i, j] = similarity between sentence i and j
        sim = self._cosine_sim_matrix(emb)

        # ========================================================================
        # STEP 4: Initialize Clustering State
        # ========================================================================
        # Purpose: Track which sentences still need to be assigned to clusters
        #
        # Processing Logic:
        #   1. Create set of remaining unassigned sentence indices (0 to n-1)
        #      - Initially contains all sentence indices
        #      - Will be reduced as sentences are assigned to clusters
        #   2. Initialize empty clusters list
        #      - Will accumulate ClusterInternal objects as clusters are formed
        n = len(items)
        remaining: Set[int] = set(range(n))
        clusters: List[ClusterInternal] = []

        # ========================================================================
        # STEP 5: Greedy Cluster Formation (Main Loop)
        # ========================================================================
        # Purpose: Iteratively form clusters by selecting seeds and grouping similar sentences
        #
        # Processing Logic:
        #   While unassigned sentences remain AND cluster count < max_clusters:
        #     - Select deterministic seed (most central sentence)
        #     - Group all sentences with similarity >= threshold to seed
        #     - Record cluster and update remaining set
        while remaining and len(clusters) < self.config.cluster_max_clusters:
            remaining_list = sorted(remaining)

            # Step 5.1: Deterministic Seed Selection
            # Purpose: Choose the most "central" sentence as cluster representative
            #
            # Processing Logic:
            #   1. Sort remaining indices for stable ordering (deterministic behavior)
            #   2. For each remaining sentence, compute total similarity to all other remaining sentences
            #      - seed_score(i) = sum of similarities from sentence i to all remaining sentences
            #   3. Select sentence with highest total similarity as cluster seed
            #      - This ensures the seed is the most "central" or representative sentence
            def seed_score(i: int) -> float:
                return float(sim[i, remaining_list].sum())

            seed = max(remaining_list, key=seed_score)

            # Step 5.2: Threshold-Based Grouping
            # Purpose: Group semantically similar sentences together
            #
            # Processing Logic:
            #   1. Find all remaining sentences with similarity >= threshold to the seed
            #      - Iterate through all remaining sentences
            #      - Check if sim[seed, j] >= cluster_similarity_threshold
            #   2. These sentences form the new cluster
            #      - All members are semantically similar to the seed and to each other (transitive)
            #   3. Remove all cluster members from remaining set
            #      - Prevents sentences from being assigned to multiple clusters
            members = [
                j
                for j in remaining_list
                if float(sim[seed, j]) >= self.config.cluster_similarity_threshold
            ]
            for j in members:
                remaining.discard(j)

            # Step 5.3: Update State
            # Purpose: Record cluster membership and associated comment IDs
            #
            # Processing Logic:
            #   1. Extract comment_ids from all cluster members
            #   2. Deduplicate comment_ids (multiple sentences can share same comment_id)
            #   3. Sort comment_ids for stable ordering (deterministic output)
            #   4. Create ClusterInternal object with:
            #      - member_indices: list of sentence indices in the cluster
            #      - comment_ids: deduplicated, sorted list of associated comment IDs
            #   5. Append to clusters list
            comment_ids = sorted({items[j].comment_id for j in members})
            clusters.append(ClusterInternal(member_indices=members, comment_ids=comment_ids))

        # ========================================================================
        # STEP 6: Handle Leftover Sentences
        # ========================================================================
        # Purpose: Ensure all sentences are assigned to clusters, even if max_clusters limit reached
        #
        # Processing Logic:
        #   1. For any remaining unassigned sentences (after max_clusters reached):
        #      - Create singleton clusters (one sentence per cluster)
        #      - Each singleton contains only one sentence
        #   2. Process in sorted order for determinism
        #      - Ensures consistent output when max_clusters limit is hit
        for i in sorted(remaining):
            clusters.append(ClusterInternal(member_indices=[i], comment_ids=[items[i].comment_id]))

        # ========================================================================
        # STEP 7: Stable Output Ordering
        # ========================================================================
        # Purpose: Ensure deterministic output order for testability and reproducibility
        #
        # Processing Logic:
        #   1. Sort clusters by:
        #      - Priority 1: Cluster size (descending): larger clusters first
        #      - Priority 2: First comment_id (ascending): stable tie-breaking
        #   2. This ensures consistent output ordering across runs
        #      - Important for testing and reproducible results
        clusters.sort(key=lambda c: (-len(c.member_indices), c.comment_ids[0]))
        return clusters


def cluster_sentences_greedy_threshold(
    items: List[SentenceItem],
    config: AppConfig
) -> List[ClusterInternal]:
    log_info(
        "clustering.greedy_threshold",
        item_count=len(items),
        similarity_threshold=config.cluster_similarity_threshold,
        max_clusters=config.cluster_max_clusters,
        embedding_provider=config.embedding_provider,
    )
    return GreedyThresholdClusterer(config).cluster(items)
