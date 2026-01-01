from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import AppConfig
from src.logging_utils import log_info
from src.pipeline.clustering import ClusterInternal
from src.pipeline.pipeline import ClusterReport, SemanticSentence, stable_dedupe_sorted
from src.pipeline.sentiment import cluster_sentiment


@dataclass(frozen=True, slots=True)
class Aggregator:
    config: AppConfig

    def build_reports(
        self,
        *,
        theme: str,
        sentences: List[SemanticSentence],
        clusters: List[ClusterInternal],
        titles: List[str],
    ) -> List[ClusterReport]:
        log_info(
            "aggregate.build_reports",
            cluster_count=len(clusters),
            sentence_count=len(sentences),
            representative_cap=int(self.config.llm_representative_sentences_per_cluster),
        )
        """
        Convert sentence-level clusters to comment-ID-level reports with sentiment, representative texts, etc.
        
        Purpose: Map sentence-level clusters to comment-ID-level reporting view. This aggregation step
        converts the internal clustering representation (based on sentence indices) into a reporting
        format that groups by comment IDs and provides cluster-level sentiment, representative texts,
        and other metadata needed for final output generation.
        
        Input Format:
          - theme: str, theme string (currently unused but kept for API consistency)
          - sentences: List[SemanticSentence], all sentences with sentiment scores
            Each contains: {sentence_id: str, comment_id: str, text: str, source: str, compound: float}
          - clusters: List[ClusterInternal], sentence-level clusters
            Each contains: {member_indices: List[int], comment_ids: List[str]}
          - titles: List[str], pre-generated titles for each cluster
        
        Output Format:
          - List[ClusterReport], comment-ID-level reports
            Each contains: {
              title: str,                    # Pre-generated cluster title
              sentiment: SentimentLabel,      # "positive" | "neutral" | "negative"
              baseline_comment_ids: tuple,   # Baseline comment ID list (deduplicated, sorted)
              comparison_comment_ids: tuple, # Comparison comment ID list (deduplicated, sorted)
              baseline_representative_texts: tuple,  # Baseline representative texts (max N items)
              comparison_representative_texts: tuple # Comparison representative texts (max N items)
            }
        """
        # ========================================================================
        # STEP 1: Initialize Output Container
        # ========================================================================
        # Purpose: Prepare list to accumulate cluster reports
        reports: List[ClusterReport] = []
        
        # ========================================================================
        # STEP 2: Process Each Cluster
        # ========================================================================
        # Purpose: Convert each sentence-level cluster into a comment-ID-level report
        for idx, c in enumerate(clusters):
            # Step 2.1: Collect Member Sentences
            # Purpose: Retrieve all SemanticSentence objects belonging to this cluster
            #
            # Processing Logic:
            #   - Use member_indices from ClusterInternal to index into sentences list
            #   - Collect all sentences that are members of this cluster
            #   - Maintains deterministic order based on member_indices
            members = [sentences[i] for i in c.member_indices]

            # Step 2.2: Group Comment IDs by Source
            # Purpose: Separate and deduplicate comment IDs by baseline/comparison source
            #
            # Processing Logic:
            #   1. Filter members by source ("baseline" or "comparison")
            #   2. Extract comment_id from each filtered member
            #   3. Use stable_dedupe_sorted to:
            #      - Remove duplicate comment IDs (multiple sentences can share same comment_id)
            #      - Sort comment IDs for stable, deterministic ordering
            baseline_ids = stable_dedupe_sorted([m.comment_id for m in members if m.source == "baseline"])
            comparison_ids = stable_dedupe_sorted([m.comment_id for m in members if m.source == "comparison"])

            # Step 2.3: Aggregate Sentiment Scores
            # Purpose: Compute cluster-level sentiment label from sentence-level sentiment scores
            #
            # Processing Logic:
            #   1. Extract compound scores from all member sentences
            #   2. Pass to cluster_sentiment() which applies aggregation rules:
            #      - If any sentence has compound < strong_negative_threshold: "negative"
            #      - Otherwise count positive/negative sentences, majority wins
            #      - If tie or no clear tendency: "neutral"
            compounds = [m.compound for m in members]
            sent_label = cluster_sentiment(compounds, self.config)

            # Step 2.4: Select Representative Texts
            # Purpose: Choose first N sentences per cohort as representative examples
            #
            # Processing Logic:
            #   1. Initialize empty lists for baseline and comparison representative texts
            #   2. Get cap limit from config (llm_representative_sentences_per_cluster)
            #   3. Iterate through member_indices in order (deterministic):
            #      - For each sentence, check its source
            #      - If source is "baseline" and baseline_rep not at cap: append text
            #      - If source is "comparison" and comparison_rep not at cap: append text
            #   4. Maintains deterministic order by processing member_indices sequentially
            #   5. Ensures at most N representative texts per cohort
            baseline_rep: list[str] = []
            comparison_rep: list[str] = []
            cap = int(self.config.llm_representative_sentences_per_cluster)
            for i in c.member_indices:
                s = sentences[i]
                if s.source == "baseline" and len(baseline_rep) < cap:
                    baseline_rep.append(s.text)
                if s.source == "comparison" and len(comparison_rep) < cap:
                    comparison_rep.append(s.text)

            # Step 2.5: Create ClusterReport Object
            # Purpose: Package all aggregated information into a ClusterReport
            #
            # Processing Logic:
            #   1. Create ClusterReport with:
            #      - title: Pre-generated title from titles list (by cluster index)
            #      - sentiment: Aggregated sentiment label from Step 2.3
            #      - baseline_comment_ids: Deduplicated, sorted baseline comment IDs (tuple)
            #      - comparison_comment_ids: Deduplicated, sorted comparison comment IDs (tuple)
            #      - baseline_representative_texts: Representative texts from baseline cohort (tuple)
            #      - comparison_representative_texts: Representative texts from comparison cohort (tuple)
            #   2. Convert lists to tuples for immutability (frozen dataclass requirement)
            #   3. Append to reports list
            reports.append(
                ClusterReport(
                    title=titles[idx],
                    sentiment=sent_label,
                    baseline_comment_ids=tuple(baseline_ids),
                    comparison_comment_ids=tuple(comparison_ids),
                    baseline_representative_texts=tuple(baseline_rep),
                    comparison_representative_texts=tuple(comparison_rep),
                )
            )
        
        # ========================================================================
        # STEP 3: Return Reports
        # ========================================================================
        # Purpose: Return complete list of cluster reports
        return reports


