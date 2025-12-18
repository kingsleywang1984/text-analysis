from dataclasses import replace

import pytest

from src.config import load_config
from src.models.schemas import ClusterLabeling, ComparisonSummary
from src.models.validators import (
    validate_cluster_labeling_budget,
    validate_comparison_budget,
)


def test_cluster_labeling_budget_accepts_within_range():
    cfg = load_config()

    # within default range (e.g. 2-3)
    label = ClusterLabeling(
        title="Login issues",
        key_insights=["Insight A", "Insight B"],
    )
    validate_cluster_labeling_budget(label, cfg)

    label2 = ClusterLabeling(
        title="Login issues",
        key_insights=["Insight A", "Insight B", "Insight C"],
    )
    validate_cluster_labeling_budget(label2, cfg)


def test_cluster_labeling_budget_rejects_outside_range():
    cfg = load_config()

    too_few = ClusterLabeling(
        title="Login issues",
        key_insights=["Only one"],
    )
    with pytest.raises(ValueError, match="key_insights count"):
        validate_cluster_labeling_budget(too_few, cfg)

    too_many = ClusterLabeling(
        title="Login issues",
        key_insights=["1", "2", "3", "4"],
    )
    with pytest.raises(ValueError, match="key_insights count"):
        validate_cluster_labeling_budget(too_many, cfg)


def test_comparison_budget_accepts_within_range():
    cfg = load_config()

    summary = ComparisonSummary(
        key_similarities=["Both mention login issues."],
        key_differences=["Baseline has more payment-related complaints."],
    )
    validate_comparison_budget(summary, cfg)


def test_comparison_budget_rejects_outside_range():
    cfg = load_config()

    too_few_similarities = ComparisonSummary(
        key_similarities=[],
        key_differences=["Some difference"],
    )
    with pytest.raises(ValueError, match="key_similarities count"):
        validate_comparison_budget(too_few_similarities, cfg)

    too_few_differences = ComparisonSummary(
        key_similarities=["Some similarity"],
        key_differences=[],
    )
    with pytest.raises(ValueError, match="key_differences count"):
        validate_comparison_budget(too_few_differences, cfg)


def test_budgets_are_configurable_via_env_test_values():
    """
    This test proves the budget logic is config-driven.
    We do NOT mutate environment here; instead we create a config variant.
    """
    base_cfg = load_config()
    cfg = replace(base_cfg, cluster_insights_min=3, cluster_insights_max=3)

    label = ClusterLabeling(
        title="Login issues",
        key_insights=["A", "B", "C"],
    )
    validate_cluster_labeling_budget(label, cfg)

    label_bad = ClusterLabeling(
        title="Login issues",
        key_insights=["A", "B"],
    )
    with pytest.raises(ValueError, match="key_insights count"):
        validate_cluster_labeling_budget(label_bad, cfg)
