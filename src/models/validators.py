from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig
from src.models.schemas import ClusterLabeling, ComparisonSummary


@dataclass(frozen=True, slots=True)
class BudgetValidator:
    """
    Budget validations for LLM outputs.

    Module-level functions remain for backward compatibility.
    """

    @staticmethod
    def validate_cluster_labeling_budget(label: ClusterLabeling, cfg: AppConfig) -> None:
        n = len(label.key_insights)
        if not (cfg.cluster_insights_min <= n <= cfg.cluster_insights_max):
            raise ValueError(
                f"key_insights count {n} out of range "
                f"[{cfg.cluster_insights_min}, {cfg.cluster_insights_max}]"
            )

    @staticmethod
    def validate_comparison_budget(summary: ComparisonSummary, cfg: AppConfig) -> None:
        ns = len(summary.key_similarities)
        nd = len(summary.key_differences)

        if not (cfg.comparison_similarities_min <= ns <= cfg.comparison_similarities_max):
            raise ValueError(
                f"key_similarities count {ns} out of range "
                f"[{cfg.comparison_similarities_min}, {cfg.comparison_similarities_max}]"
            )

        if not (cfg.comparison_differences_min <= nd <= cfg.comparison_differences_max):
            raise ValueError(
                f"key_differences count {nd} out of range "
                f"[{cfg.comparison_differences_min}, {cfg.comparison_differences_max}]"
            )


def validate_cluster_labeling_budget(label: ClusterLabeling, cfg: AppConfig) -> None:
    BudgetValidator.validate_cluster_labeling_budget(label, cfg)


def validate_comparison_budget(summary: ComparisonSummary, cfg: AppConfig) -> None:
    BudgetValidator.validate_comparison_budget(summary, cfg)


