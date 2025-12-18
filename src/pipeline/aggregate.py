from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import AppConfig
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
        """
        Map sentence-level clusters to comment-id-level reporting view.
        """
        reports: List[ClusterReport] = []
        for idx, c in enumerate(clusters):
            members = [sentences[i] for i in c.member_indices]

            baseline_ids = stable_dedupe_sorted([m.comment_id for m in members if m.source == "baseline"])
            comparison_ids = stable_dedupe_sorted([m.comment_id for m in members if m.source == "comparison"])

            compounds = [m.compound for m in members]
            sent_label = cluster_sentiment(compounds, self.config)

            # Representative texts (deterministic): first N per cohort in member_indices order
            baseline_rep: list[str] = []
            comparison_rep: list[str] = []
            cap = int(self.config.llm_representative_sentences_per_cluster)
            for i in c.member_indices:
                s = sentences[i]
                if s.source == "baseline" and len(baseline_rep) < cap:
                    baseline_rep.append(s.text)
                if s.source == "comparison" and len(comparison_rep) < cap:
                    comparison_rep.append(s.text)

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
        return reports


