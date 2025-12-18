from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config import AppConfig
from src.llm.client import LLMClient
from src.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponseComparison,
    AnalyzeResponseStandalone,
    ComparisonCluster,
    StandaloneCluster,
)
from src.models.validators import (
    validate_cluster_labeling_budget,
    validate_comparison_budget,
)
from src.pipeline.aggregate import Aggregator
from src.pipeline.clustering import (
    ClusterInternal,
    SentenceItem,
    cluster_sentences_greedy_threshold,
)
from src.pipeline.insights_tfidf import DeterministicInsightGenerator
from src.pipeline.normalize import normalize_text
from src.pipeline.pipeline import SemanticSentence, make_sentence_ids
from src.pipeline.sentiment import sentence_compound


@dataclass(frozen=True, slots=True)
class RequestAnalyzer:
    """
    Task analyzer. Module-level `analyze_request` remains as the public entrypoint.
    """

    config: AppConfig
    llm: Optional[LLMClient] = None

    @staticmethod
    def _fallback_title(theme: str, cluster_index: int, is_other: bool = False) -> str:
        if is_other:
            return "Other"
        return f"{theme} cluster {cluster_index + 1}"

    @staticmethod
    def _fallback_insights(theme: str, sentiment: str) -> List[str]:
        # Deterministic placeholder until LLM step (must be 2–3)
        return [
            f"Users report issues related to {theme}.",
            f"Overall sentiment appears {sentiment} for this cluster.",
        ]

    def _select_top_clusters_with_other(self, clusters: List[ClusterInternal]) -> List[ClusterInternal]:
        """
        Apply max cluster budget. Default strategy: OTHER.
        - Sort clusters by importance:
            1) unique comment_ids desc
            2) member count desc
            3) first comment_id asc (stable)
        - If <= max: return as-is
        - If overflow and strategy OTHER:
            return top (max-1) + one merged 'Other' cluster
        - If DROP:
            return top (max)
        """
        if not clusters:
            return []

        # deterministic importance ordering
        clusters_sorted = sorted(
            clusters,
            key=lambda c: (-len(c.comment_ids), -len(c.member_indices), c.comment_ids[0]),
        )

        max_n = self.config.cluster_max_clusters
        if len(clusters_sorted) <= max_n:
            return clusters_sorted

        if self.config.cluster_overflow_strategy == "DROP":
            return clusters_sorted[:max_n]

        # OTHER (default)
        head = clusters_sorted[: max_n - 1]
        tail = clusters_sorted[max_n - 1 :]

        # merge tail into one cluster
        merged_member_indices = []
        merged_comment_ids = set()
        for c in tail:
            merged_member_indices.extend(c.member_indices)
            merged_comment_ids.update(c.comment_ids)

        other = ClusterInternal(
            member_indices=sorted(set(merged_member_indices)),
            comment_ids=sorted(merged_comment_ids),
        )
        return head + [other]

    def analyze(self, req: AnalyzeRequest):
        """
        Produce task-shaped response.
        - Clustering & sentiment are deterministic.
        - Title/insights are deterministic placeholders (LLM comes later).
        """
        # ========================================================================
        # STEP 1: Text Normalization and Input Data Preparation
        # ========================================================================
        # Purpose: Normalize and convert input baseline and comparison sentences into internal data structures
        # 
        # Input Format:
        #   - req.baseline: List[InputSentence], each containing {sentence: str, id: str}
        #   - req.comparison: Optional[List[InputSentence]], optional
        #
        # Processing Logic:
        #   1. Normalize each sentence using normalize_text():
        #      - Unicode normalization (NFKC)
        #      - Trim leading/trailing whitespace
        #      - Collapse consecutive whitespace to single spaces
        #   2. Filter out sentences that become empty strings after normalization
        #   3. Process in order: baseline first, then comparison (ensures deterministic ordering)
        #   4. Create SentenceItem for each valid sentence, marking source (baseline/comparison)
        #
        # Output Format:
        #   - raw_items: List[SentenceItem], each containing {text: str, comment_id: str, source: str}
        #     Example: [SentenceItem(text="Withholding my money", comment_id="uuid-123", source="baseline"), ...]
        raw_items: List[SentenceItem] = []
        raw_comment_ids: List[str] = []
        raw_sources: List[str] = []
        for s in req.baseline:
            t = normalize_text(s.sentence)
            if not t:
                continue
            raw_items.append(SentenceItem(text=t, comment_id=s.id, source="baseline"))
            raw_comment_ids.append(s.id)
            raw_sources.append("baseline")

        if req.comparison:
            for s in req.comparison:
                t = normalize_text(s.sentence)
                if not t:
                    continue
                raw_items.append(SentenceItem(text=t, comment_id=s.id, source="comparison"))
                raw_comment_ids.append(s.id)
                raw_sources.append("comparison")

        if not raw_items:
            raise RuntimeError("No valid sentences after normalization")

        # ========================================================================
        # STEP 2: Sentence-Level Sentiment Analysis
        # ========================================================================
        # Purpose: Compute sentiment scores for each sentence, used for cluster-level sentiment aggregation
        #
        # Input Format:
        #   - raw_items: List[SentenceItem], from STEP 1
        #
        # Processing Logic:
        #   1. Generate deterministic IDs for each sentence (s0, s1, s2, ...)
        #   2. Use VADER sentiment analyzer to compute compound score for each sentence
        #      - compound range: [-1.0, 1.0]
        #      - Negative values indicate negative sentiment, positive values indicate positive sentiment
        #   3. Create SemanticSentence objects containing sentence info and sentiment score
        #
        # Output Format:
        #   - semantic_sentences: List[SemanticSentence]
        #     Each contains: {sentence_id: str, comment_id: str, text: str, source: str, compound: float}
        #     Example: [SemanticSentence(sentence_id="s0", comment_id="uuid-123", text="...", source="baseline", compound=-0.6249), ...]
        sentence_ids = make_sentence_ids(len(raw_items))
        semantic_sentences: List[SemanticSentence] = []
        for i, it in enumerate(raw_items):
            semantic_sentences.append(
                SemanticSentence(
                    sentence_id=sentence_ids[i],
                    comment_id=it.comment_id,
                    text=it.text,
                    source=it.source,
                    compound=sentence_compound(it.text),
                )
            )

        # ========================================================================
        # STEP 3: Similarity-Based Greedy Clustering
        # ========================================================================
        # Purpose: Group semantically similar sentences into the same cluster
        #
        # Input Format:
        #   - raw_items: List[SentenceItem], from STEP 1
        #   - self.config: Contains clustering threshold (cluster_similarity_threshold) and other config
        #
        # Processing Logic:
        #   1. Use EmbeddingFactory to generate sentence embedding vectors (TF-IDF or OpenAI embedding)
        #   2. Compute cosine similarity matrix between sentences
        #   3. Greedy clustering algorithm:
        #      - Select sentence with highest total similarity to remaining sentences as seed
        #      - Group all sentences with similarity >= threshold to seed into same cluster
        #      - Repeat until all sentences are assigned or max clusters reached
        #      - Remaining sentences become singleton clusters
        #   4. Each cluster contains: member_indices (sentence indices) and comment_ids (deduplicated comment IDs)
        #
        # Output Format:
        #   - clusters_raw: List[ClusterInternal]
        #     Each contains: {member_indices: List[int], comment_ids: List[str]}
        #     Example: [ClusterInternal(member_indices=[0, 1, 2], comment_ids=["uuid-123", "uuid-456"]), ...]
        clusters_raw = cluster_sentences_greedy_threshold(raw_items, self.config)

        # ========================================================================
        # STEP 4: Cluster Selection and Overflow Handling
        # ========================================================================
        # Purpose: Limit cluster count based on config and handle overflow scenarios
        #
        # Input Format:
        #   - clusters_raw: List[ClusterInternal], from STEP 3
        #   - self.config.cluster_max_clusters: Maximum number of clusters
        #   - self.config.cluster_overflow_strategy: "OTHER" or "DROP"
        #
        # Processing Logic:
        #   1. Sort clusters by importance:
        #      - Priority 1: Unique comment_ids count (descending)
        #      - Priority 2: Member count (descending)
        #      - Priority 3: First comment_id (ascending, for stability)
        #   2. If cluster count <= max_clusters: return all clusters
        #   3. If overflow and strategy is "DROP": return top max_clusters
        #   4. If overflow and strategy is "OTHER": return top (max_clusters-1) + one merged "Other" cluster
        #      - "Other" cluster contains merged content from all remaining clusters
        #
        # Output Format:
        #   - clusters_internal: List[ClusterInternal], count <= max_clusters
        #     If OTHER strategy is used, the last cluster is the merged "Other" cluster
        clusters_internal = self._select_top_clusters_with_other(clusters_raw)

        # ========================================================================
        # STEP 5: Initialize Output Containers and Overflow Flag
        # ========================================================================
        # Purpose: Prepare output lists for both modes and flag whether overflow-merged "Other" cluster exists
        #
        # Output Format:
        #   - clusters_out_standalone: List[StandaloneCluster], for standalone mode
        #   - clusters_out_comparison: List[ComparisonCluster], for comparison mode
        #   - overflowed: bool, indicates if OTHER strategy was used and overflow occurred
        clusters_out_standalone: List[StandaloneCluster] = []
        clusters_out_comparison: List[ComparisonCluster] = []
        overflowed = (
            self.config.cluster_overflow_strategy == "OTHER"
            and len(clusters_raw) > self.config.cluster_max_clusters
        )

        # ========================================================================
        # STEP 6: Generate Deterministic Fallback Titles
        # ========================================================================
        # Purpose: Generate initial titles for each cluster, used for subsequent aggregation report generation
        #
        # Input Format:
        #   - clusters_internal: List[ClusterInternal], from STEP 4
        #   - req.theme: Theme string
        #   - overflowed: bool, whether overflow occurred
        #
        # Processing Logic:
        #   1. Generate fallback title for each cluster:
        #      - Regular cluster: "{theme} cluster {index+1}"
        #      - If overflow-merged last cluster: "Other"
        #   2. These titles will be replaced by LLM or TF-IDF generated titles in subsequent steps
        #
        # Output Format:
        #   - fallback_titles: List[str]
        #     Example: ["payment issues cluster 1", "refund requests cluster 2", "Other"]
        fallback_titles: List[str] = []
        for idx, c in enumerate(clusters_internal):
            is_other_cluster = overflowed and (idx == len(clusters_internal) - 1)
            fallback_titles.append(self._fallback_title(req.theme, idx, is_other=is_other_cluster))

        # ========================================================================
        # STEP 7: Generate Cluster Reports (Aggregated View)
        # ========================================================================
        # Purpose: Convert sentence-level clusters to comment-ID-level reports with sentiment, representative texts, etc.
        #
        # Input Format:
        #   - semantic_sentences: List[SemanticSentence], from STEP 2
        #   - clusters_internal: List[ClusterInternal], from STEP 4
        #   - fallback_titles: List[str], from STEP 6
        #
        # Processing Logic:
        #   1. For each cluster:
        #      - Collect all member sentences
        #      - Group comment_ids by source (baseline/comparison) and deduplicate/sort
        #      - Aggregate sentiment scores from all member sentences, compute cluster-level sentiment label
        #      - Select representative texts (first N sentences per cohort, in member_indices order)
        #   2. Sentiment aggregation rules:
        #      - If any sentence has compound < strong_negative_threshold: "negative"
        #      - Otherwise count positive/negative sentences, majority wins
        #      - If tie or no clear tendency: "neutral"
        #
        # Output Format:
        #   - reports: List[ClusterReport]
        #     Each contains: {
        #       title: str,                    # Fallback title
        #       sentiment: SentimentLabel,      # "positive" | "neutral" | "negative"
        #       baseline_comment_ids: tuple,   # Baseline comment ID list (deduplicated, sorted)
        #       comparison_comment_ids: tuple, # Comparison comment ID list (deduplicated, sorted)
        #       baseline_representative_texts: tuple,  # Baseline representative texts (max N items)
        #       comparison_representative_texts: tuple # Comparison representative texts (max N items)
        #     }
        reports = Aggregator(self.config).build_reports(
            theme=req.theme,
            sentences=semantic_sentences,
            clusters=clusters_internal,
            titles=fallback_titles,
        )

        # ========================================================================
        # STEP 8: Initialize Deterministic Insight Generator
        # ========================================================================
        # Purpose: Create TF-IDF generator for fallback title and insight generation
        #
        # Note: This generator provides deterministic fallback content when LLM is unavailable or fails
        fallback = DeterministicInsightGenerator(self.config)

        # ========================================================================
        # STEP 9: Generate Final Output for Each Cluster
        # ========================================================================
        # Purpose: Generate corresponding output clusters based on request type (standalone/comparison)
        #
        # Processing Logic: Iterate through each cluster and corresponding report to generate final output
        for idx, (c, report) in enumerate(zip(clusters_internal, reports)):
            texts = list(report.baseline_representative_texts or report.comparison_representative_texts)
            sent = report.sentiment

            is_other_cluster = overflowed and (idx == len(clusters_internal) - 1)

            title = report.title
            if req.comparison:
                # ================================================================
                # COMPARISON Mode Processing
                # ================================================================
                # Purpose: Generate comparison mode output with baseline and comparison contrast information
                #
                # Output Requirements:
                #   - Must include: title, sentiment, baselineSentences, comparisonSentences, 
                #                  keySimilarities, keyDifferences
                #   - Must NOT include: keyInsights
                #
                # Step 9.1: Extract comment ID lists
                # Output: baseline_ids, comparison_ids (List[str])
                baseline_ids = list(report.baseline_comment_ids)
                comparison_ids = list(report.comparison_comment_ids)

                # Step 9.2: Validate comparison validity
                # Requirement: Both baseline and comparison must have content, otherwise skip this cluster
                if not baseline_ids or not comparison_ids:
                    continue

                # Step 9.3: Generate cluster title (LLM-first strategy)
                # Processing Logic:
                #   1. First generate fallback title using TF-IDF (based on representative texts from both cohorts)
                #   2. If LLM is available and cluster index < llm_max_clusters and not "Other" cluster:
                #      - Call LLM's label_cluster method to generate title
                #      - Validate output meets budget requirements (length limits, etc.)
                #      - If successful, use LLM-generated title; if failed, use fallback title
                # Output: title (str), 3-80 character cluster title
                title = fallback.comparison_title(
                    theme=req.theme,
                    baseline_texts=list(report.baseline_representative_texts),
                    comparison_texts=list(report.comparison_representative_texts),
                )
                if self.llm is not None and (idx < int(self.config.llm_max_clusters)) and not is_other_cluster:
                    try:
                        title_label = self.llm.label_cluster(
                            req.theme,
                            (list(report.baseline_representative_texts) + list(report.comparison_representative_texts))[
                                : int(self.config.llm_representative_sentences_per_cluster)
                            ],
                        )
                        validate_cluster_labeling_budget(title_label, self.config)
                        title = title_label.title
                    except Exception:
                        pass

                # Step 9.4: Generate similarities and differences (deterministic fallback)
                # Processing Logic:
                #   1. Use TF-IDF to extract keywords from both cohorts
                #   2. Compute shared keywords, baseline-only keywords, comparison-only keywords
                #   3. Generate templated similarity and difference descriptions based on keywords
                #   4. Ensure output count meets configured min/max requirements
                # Output: key_sim (List[str]), key_diff (List[str])
                key_sim, key_diff = fallback.comparison_similarities_differences(
                    theme=req.theme,
                    cluster_title=title,
                    baseline_texts=list(report.baseline_representative_texts),
                    comparison_texts=list(report.comparison_representative_texts),
                )

                # Step 9.5: LLM enhancement for similarities and differences (optional)
                # Processing Logic:
                #   1. If LLM is available and cluster index < llm_max_clusters:
                #      - Call LLM's summarize_cluster_comparison method
                #      - Pass theme, title, sentiment, representative texts from both cohorts
                #      - Validate output meets budget requirements
                #      - If successful, replace fallback content with LLM-generated content
                #   2. If LLM call fails, continue using fallback content
                # Output: key_sim, key_diff (List[str]), may be enhanced by LLM
                if self.llm is not None and idx < int(self.config.llm_max_clusters):
                    try:
                        baseline_rep = list(report.baseline_representative_texts)[: int(self.config.llm_representative_sentences_per_cluster)]
                        comparison_rep = list(report.comparison_representative_texts)[: int(self.config.llm_representative_sentences_per_cluster)]
                        summary = self.llm.summarize_cluster_comparison(
                            theme=req.theme,
                            cluster_title=title,
                            sentiment=sent,
                            baseline_sentences=baseline_rep,
                            comparison_sentences=comparison_rep,
                        )
                        validate_comparison_budget(summary, self.config)
                        key_sim = summary.key_similarities
                        key_diff = summary.key_differences
                    except Exception:
                        pass

                # Step 9.6: Build ComparisonCluster output object
                # Output Format: ComparisonCluster {
                #   title: str,                    # Cluster title
                #   sentiment: SentimentLabel,       # Sentiment label
                #   baselineSentences: List[str],   # Baseline comment ID list
                #   comparisonSentences: List[str], # Comparison comment ID list
                #   keySimilarities: List[str],     # Similarity description list
                #   keyDifferences: List[str]       # Difference description list
                # }
                clusters_out_comparison.append(
                    ComparisonCluster(
                        title=title,
                        sentiment=sent,
                        baselineSentences=baseline_ids,
                        comparisonSentences=comparison_ids,
                        keySimilarities=key_sim,
                        keyDifferences=key_diff,
                    )
                )
            else:
                # ================================================================
                # STANDALONE Mode Processing
                # ================================================================
                # Purpose: Generate standalone mode output with title, sentiment, and key insights
                #
                # Output Requirements:
                #   - Must include: title, sentiment, keyInsights
                #   - Must NOT include: baselineSentences, comparisonSentences, keySimilarities, keyDifferences
                #
                # Step 9.1: Generate title and insights (deterministic fallback)
                # Processing Logic:
                #   1. Use TF-IDF to extract keywords (top 4)
                #   2. Generate title using first 2 keywords
                #   3. Generate 2-3 templated insights (meeting configured min/max requirements)
                # Output: title (str), insights (List[str], 2-3 items)
                title, insights = fallback.standalone_title_and_insights(
                    theme=req.theme,
                    sentiment=sent,
                    texts=texts,
                )

                # Step 9.2: LLM enhancement for title and insights (LLM-first strategy)
                # Processing Logic:
                #   1. If LLM is available and cluster index < llm_max_clusters and not "Other" cluster:
                #      - Call LLM's label_cluster method
                #      - Pass theme and first 50 representative texts (limit prompt size)
                #      - Validate output meets budget requirements
                #      - If successful, use LLM-generated title and insights; if failed, use fallback content
                # Output: title (str), insights (List[str]), may be enhanced by LLM
                if self.llm is not None and (idx < int(self.config.llm_max_clusters)) and not is_other_cluster:
                    try:
                        labeling = self.llm.label_cluster(req.theme, texts[:50])  # cap prompt size
                        validate_cluster_labeling_budget(labeling, self.config)
                        title = labeling.title
                        insights = labeling.key_insights
                    except Exception:
                        # Any failure => fallback
                        pass

                # Step 9.3: Build StandaloneCluster output object
                # Output Format: StandaloneCluster {
                #   title: str,              # Cluster title
                #   sentiment: SentimentLabel, # Sentiment label
                #   keyInsights: List[str]    # Key insights list (2-3 items)
                # }
                clusters_out_standalone.append(
                    StandaloneCluster(
                        title=title,
                        sentiment=sent,
                        keyInsights=insights,
                    )
                )

        # ========================================================================
        # STEP 10: Validate and Return Final Result
        # ========================================================================
        # Purpose: Validate output validity and return corresponding response type
        #
        # Processing Logic:
        #   1. If comparison mode:
        #      - Validate at least one valid comparison cluster exists (both baseline and comparison have content)
        #      - If not, raise exception
        #      - Return AnalyzeResponseComparison
        #   2. If standalone mode:
        #      - Directly return AnalyzeResponseStandalone
        #
        # Final Output Format:
        #   - AnalyzeResponseComparison {
        #       clusters: List[ComparisonCluster]  # At least 1 cluster
        #     }
        #   or
        #   - AnalyzeResponseStandalone {
        #       clusters: List[StandaloneCluster]   # At least 1 cluster
        #     }
        if req.comparison:
            if not clusters_out_comparison:
                raise RuntimeError("No comparable clusters found (baseline/comparison did not overlap after processing)")
            return AnalyzeResponseComparison(clusters=clusters_out_comparison)

        return AnalyzeResponseStandalone(clusters=clusters_out_standalone)


def analyze_request(req: AnalyzeRequest, config: AppConfig, llm: Optional[LLMClient] = None):
    return RequestAnalyzer(config=config, llm=llm).analyze(req)
