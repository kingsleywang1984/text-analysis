from __future__ import annotations

from dataclasses import dataclass
from typing import List

SYSTEM_PRODUCT_ANALYST = (
    "You are a product insights analyst.\n"
    "Follow instructions exactly.\n"
    "Return ONLY valid JSON. Do not wrap in markdown. Do not include explanations.\n"
)


@dataclass(frozen=True, slots=True)
class PromptBuilder:
    """
    Centralize prompt construction. Module-level functions remain for backward compatibility.
    """

    @staticmethod
    def user_label_cluster(theme: str, sentences: List[str]) -> str:
        # NOTE: budgets are enforced by validators (config-driven), not hardcoded here.
        return (
            f"Theme: {theme}\n"
            "Task: Create a concise cluster title and key insights.\n"
            'Return JSON exactly in this shape: {"title": "...", "key_insights": ["...", "..."]}\n'
            "Rules:\n"
            "- Title: 3-80 chars, concise, specific\n"
            "- Insights: short bullet-like sentences\n\n"
            "Sentences:\n- " + "\n- ".join(sentences)
        )

    @staticmethod
    def user_summarize_comparison(theme: str, baseline_titles: List[str], comparison_titles: List[str]) -> str:
        return (
            f"Theme: {theme}\n"
            "Task: Compare baseline vs comparison cluster titles.\n"
            'Return JSON exactly in this shape: {"key_similarities": ["..."], "key_differences": ["..."]}\n'
            "Rules:\n"
            "- Similarities/differences should be concise and actionable\n\n"
            "Baseline cluster titles:\n- " + "\n- ".join(baseline_titles) +
            "\n\nComparison cluster titles:\n- " + "\n- ".join(comparison_titles)
        )

    @staticmethod
    def user_summarize_cluster_comparison(
        *,
        theme: str,
        cluster_title: str,
        sentiment: str,
        baseline_sentences: List[str],
        comparison_sentences: List[str],
    ) -> str:
        return (
            f"Theme: {theme}\n"
            f"Cluster title: {cluster_title}\n"
            f"Cluster sentiment: {sentiment}\n"
            "Task: For THIS cluster only, compare baseline vs comparison feedback.\n"
            'Return JSON exactly in this shape: {"key_similarities": ["..."], "key_differences": ["..."]}\n'
            "Rules:\n"
            "- IMPORTANT: Your similarities/differences MUST be consistent with the provided sentiment label.\n"
            "  Do NOT claim the sentiment is positive if sentiment is negative (and vice versa).\n"
            "- Similarities: what both cohorts express in common\n"
            "- Differences: what changes between cohorts (volume, details, phrasing)\n"
            "- Keep items concise and actionable\n\n"
            "Baseline representative sentences:\n- " + "\n- ".join(baseline_sentences) +
            "\n\nComparison representative sentences:\n- " + "\n- ".join(comparison_sentences)
        )


def user_label_cluster(theme: str, sentences: List[str]) -> str:
    return PromptBuilder.user_label_cluster(theme, sentences)


def user_summarize_comparison(theme: str, baseline_titles: List[str], comparison_titles: List[str]) -> str:
    return PromptBuilder.user_summarize_comparison(theme, baseline_titles, comparison_titles)


def user_summarize_cluster_comparison(
    *,
    theme: str,
    cluster_title: str,
    sentiment: str,
    baseline_sentences: List[str],
    comparison_sentences: List[str],
) -> str:
    return PromptBuilder.user_summarize_cluster_comparison(
        theme=theme,
        cluster_title=cluster_title,
        sentiment=sentiment,
        baseline_sentences=baseline_sentences,
        comparison_sentences=comparison_sentences,
    )
