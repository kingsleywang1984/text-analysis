from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import AppConfig


def _top_terms_tfidf(texts: Sequence[str], *, top_k: int) -> List[str]:
    """
    Deterministic keyword extraction using TF-IDF:
    - Fit a vectorizer on the given texts
    - Rank terms by mean TF-IDF across documents
    """
    if not texts:
        return []

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X = vec.fit_transform(list(texts))
    if X.shape[1] == 0:
        return []

    # mean score per term (stable)
    scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: (-float(x[1]), x[0]))
    return [t for t, _ in ranked[:top_k]]


def _cap_list(items: List[str], n: int) -> List[str]:
    if len(items) >= n:
        return items[:n]
    return items + ["(no additional insight)"] * (n - len(items))


@dataclass(frozen=True, slots=True)
class DeterministicInsightGenerator:
    """
    Deterministic fallback generators (TF-IDF/keyword + templates).
    Intended as the fallback path when LLM is unavailable or fails.
    """

    cfg: AppConfig

    def standalone_title_and_insights(self, *, theme: str, sentiment: str, texts: Sequence[str]) -> tuple[str, List[str]]:
        terms = _top_terms_tfidf(texts, top_k=4)
        title = " / ".join(terms[:2]) if terms else f"{theme} insights"
        # ensure budgets (2-3)
        n = max(self.cfg.cluster_insights_min, 2)
        n = min(n, self.cfg.cluster_insights_max)
        insights = [
            f"Key theme: **{theme}**; cluster sentiment appears **{sentiment}**.",
            f"Top terms: **{', '.join(terms[:3])}**." if terms else "Users share feedback on this theme.",
            "Details vary across comments; consider investigating representative examples.",
        ]
        insights = _cap_list([s for s in insights if s and s.strip()], n)
        return title, insights

    def comparison_similarities_differences(
        self,
        *,
        theme: str,
        cluster_title: str,
        baseline_texts: Sequence[str],
        comparison_texts: Sequence[str],
    ) -> tuple[List[str], List[str]]:
        # Extract top terms per cohort
        base_terms = _top_terms_tfidf(baseline_texts, top_k=6)
        comp_terms = _top_terms_tfidf(comparison_texts, top_k=6)

        shared = sorted(set(base_terms) & set(comp_terms))
        base_only = sorted(set(base_terms) - set(comp_terms))
        comp_only = sorted(set(comp_terms) - set(base_terms))

        ns = max(self.cfg.comparison_similarities_min, 1)
        ns = min(ns, self.cfg.comparison_similarities_max)
        nd = max(self.cfg.comparison_differences_min, 1)
        nd = min(nd, self.cfg.comparison_differences_max)

        sims: List[str] = [
            f"Both cohorts discuss **{cluster_title}** within theme **{theme}**.",
            f"Shared terms: **{', '.join(shared[:3])}**." if shared else "Language overlaps but with different emphasis.",
        ]
        diffs: List[str] = [
            f"Baseline unique terms: **{', '.join(base_only[:3])}**." if base_only else "Baseline has fewer unique terms.",
            f"Comparison unique terms: **{', '.join(comp_only[:3])}**." if comp_only else "Comparison has fewer unique terms.",
        ]

        sims = _cap_list([s for s in sims if s and s.strip()], ns)
        diffs = _cap_list([s for s in diffs if s and s.strip()], nd)
        return sims, diffs

    def comparison_title(self, *, theme: str, baseline_texts: Sequence[str], comparison_texts: Sequence[str]) -> str:
        texts = list(baseline_texts) + list(comparison_texts)
        terms = _top_terms_tfidf(texts, top_k=4)
        return " / ".join(terms[:2]) if terms else f"{theme} comparison"



