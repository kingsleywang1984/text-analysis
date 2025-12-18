# LLM Usage Analysis in Text Analysis Pipeline

## Overview

The LLM (Large Language Model) is used **optionally** in the pipeline to enhance output quality. It follows an **LLM-first strategy with deterministic fallback**, meaning:
- LLM is tried first when available
- If LLM fails or is unavailable, deterministic TF-IDF fallback is used
- This ensures the pipeline always produces output, even without LLM

## LLM Usage Locations

### 1. **COMPARISON Mode - Cluster Title Generation** (Line 358-369)

**Location**: `src/pipeline/analyze.py`, Step 9.3 in COMPARISON mode processing

**LLM Method**: `label_cluster(theme, sentences)`

**Conditions**:
- `self.llm is not None`
- `cluster_index < llm_max_clusters`
- Cluster is NOT the "Other" merged cluster

**Input**:
- Theme: Request theme string
- Sentences: Combined representative texts from both baseline and comparison cohorts
  - Limited to `llm_representative_sentences_per_cluster` sentences

**Output**: `ClusterLabeling` containing:
- `title`: 3-80 character cluster title (✅ **USED**)
- `key_insights`: List of insights (❌ **NOT USED** in comparison mode - ComparisonCluster doesn't have keyInsights field)

**Impact on Final Output**:
- **Affects**: `ComparisonCluster.title` **ONLY**
- **Note**: The `key_insights` from `label_cluster` response is **discarded** because comparison mode doesn't use keyInsights
- **Fallback**: TF-IDF generated title from `DeterministicInsightGenerator.comparison_title()`

**Code Reference**:
```python
if self.llm is not None and (idx < int(self.config.llm_max_clusters)) and not is_other_cluster:
    try:
        title_label = self.llm.label_cluster(
            req.theme,
            (list(report.baseline_representative_texts) + list(report.comparison_representative_texts))[
                : int(self.config.llm_representative_sentences_per_cluster)
            ],
        )
        validate_cluster_labeling_budget(title_label, self.config)
        title = title_label.title  # ⚠️ Only title is used, key_insights is discarded
    except Exception:
        pass  # Falls back to TF-IDF title
```

---

### 2. **COMPARISON Mode - Similarities and Differences** (Line 394-409)

**Location**: `src/pipeline/analyze.py`, Step 9.5 in COMPARISON mode processing

**LLM Method**: `summarize_cluster_comparison(theme, cluster_title, sentiment, baseline_sentences, comparison_sentences)`

**Conditions**:
- `self.llm is not None`
- `cluster_index < llm_max_clusters`
- (No restriction on "Other" cluster for this step)

**Input**:
- Theme: Request theme string
- Cluster title: Already generated title (may be LLM or TF-IDF)
- Sentiment: Cluster sentiment label ("positive", "neutral", "negative")
- Baseline sentences: Representative texts from baseline cohort
- Comparison sentences: Representative texts from comparison cohort
  - Both limited to `llm_representative_sentences_per_cluster` sentences

**Output**: `ComparisonSummary` containing:
- `key_similarities`: List of similarity descriptions
- `key_differences`: List of difference descriptions

**Impact on Final Output**:
- **Affects**: `ComparisonCluster.keySimilarities` and `ComparisonCluster.keyDifferences`
- **Fallback**: TF-IDF generated similarities/differences from `DeterministicInsightGenerator.comparison_similarities_differences()`

**Code Reference**:
```python
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
        pass  # Falls back to TF-IDF similarities/differences
```

---

### 3. **STANDALONE Mode - Title and Insights** (Line 460-468)

**Location**: `src/pipeline/analyze.py`, Step 9.2 in STANDALONE mode processing

**LLM Method**: `label_cluster(theme, sentences)`

**Conditions**:
- `self.llm is not None`
- `cluster_index < llm_max_clusters`
- Cluster is NOT the "Other" merged cluster

**Input**:
- Theme: Request theme string
- Sentences: Representative texts (baseline or comparison)
  - Limited to first 50 sentences to cap prompt size

**Output**: `ClusterLabeling` containing:
- `title`: 3-80 character cluster title (✅ **USED**)
- `key_insights`: List of 2-3 key insights (✅ **USED**)

**Impact on Final Output**:
- **Affects**: `StandaloneCluster.title` **AND** `StandaloneCluster.keyInsights`
- **Fallback**: TF-IDF generated title and insights from `DeterministicInsightGenerator.standalone_title_and_insights()`

**Code Reference**:
```python
if self.llm is not None and (idx < int(self.config.llm_max_clusters)) and not is_other_cluster:
    try:
        labeling = self.llm.label_cluster(req.theme, texts[:50])  # cap prompt size
        validate_cluster_labeling_budget(labeling, self.config)
        title = labeling.title        # ✅ Both title and key_insights are used
        insights = labeling.key_insights
    except Exception:
        pass  # Falls back to TF-IDF title and insights
```

---

## Impact on Final Output

### Summary Table

| Output Field | Mode | LLM Impact | Fallback Method | Notes |
|-------------|------|------------|-----------------|-------|
| `title` | COMPARISON | ✅ Enhanced by LLM | TF-IDF keywords | Only `title` from `label_cluster` is used |
| `title` | STANDALONE | ✅ Enhanced by LLM | TF-IDF keywords | Both `title` and `key_insights` from `label_cluster` are used |
| `keyInsights` | STANDALONE | ✅ Enhanced by LLM | TF-IDF templated insights | From `label_cluster.key_insights` |
| `keySimilarities` | COMPARISON | ✅ Enhanced by LLM | TF-IDF keyword-based | From `summarize_cluster_comparison` |
| `keyDifferences` | COMPARISON | ✅ Enhanced by LLM | TF-IDF keyword-based | From `summarize_cluster_comparison` |
| `sentiment` | Both | ❌ Not affected | VADER sentiment analysis | - |
| `baselineSentences` | COMPARISON | ❌ Not affected | Determined by clustering | - |
| `comparisonSentences` | COMPARISON | ❌ Not affected | Determined by clustering | - |

### Detailed Impact

#### 1. **Cluster Titles** (Both Modes)
- **Without LLM**: Generic titles like `"{theme} cluster {index+1}"` or TF-IDF keyword-based titles
- **With LLM**: Context-aware, concise titles (3-80 chars) that capture the essence of the cluster
- **Example**:
  - Fallback: `"payment issues cluster 1"` or `"money / refund"`
  - LLM: `"Payment Withholding Complaints"` or `"Refund Request Delays"`

#### 2. **Key Insights** (STANDALONE Mode Only)
- **Without LLM**: Template-based insights like:
  - `"Key theme: **payment**; cluster sentiment appears **negative**."`
  - `"Top terms: **money, refund, withdrawal**."`
- **With LLM**: Contextual, actionable insights that summarize the cluster's main points
- **Example**:
  - Fallback: Generic templated insights
  - LLM: `"Users report significant delays in receiving refunds, with some waiting over 30 days."`
- **Note**: In COMPARISON mode, `label_cluster` is called but its `key_insights` output is **discarded** because `ComparisonCluster` doesn't have a `keyInsights` field

#### 3. **Similarities and Differences** (COMPARISON Mode Only)
- **Without LLM**: Keyword-based comparisons:
  - `"Both cohorts discuss **payment issues** within theme **refunds**."`
  - `"Shared terms: **money, refund, delay**."`
- **With LLM**: Nuanced comparisons that consider sentiment and context:
  - `"Both cohorts express frustration with payment processing delays, but baseline users emphasize withdrawal restrictions while comparison users focus on refund timelines."`
  - LLM output is constrained by sentiment label to ensure consistency

### What LLM Does NOT Affect

The following pipeline steps and outputs are **completely deterministic** and not affected by LLM:

1. **Clustering** (STEP 3): Sentence grouping based on embedding similarity
2. **Sentiment Analysis** (STEP 2, 7): VADER-based sentiment scoring
3. **Cluster Selection** (STEP 4): Top-N cluster selection and overflow handling
4. **Comment ID Lists** (STEP 7, 9): Which comments belong to which clusters
5. **Representative Texts** (STEP 7): Which sentences are selected as representative
6. **Cluster Structure**: Number of clusters, cluster membership

## LLM Configuration Controls

The following configuration parameters control LLM usage:

- **`llm_provider`**: `"none"` | `"openai_compatible"` - Enables/disables LLM
- **`llm_max_clusters`**: Maximum number of clusters to enhance with LLM (default: 10)
- **`llm_representative_sentences_per_cluster`**: Number of sentences sent to LLM (default: 10)
- **`llm_timeout_seconds`**: Request timeout
- **`llm_temperature`**: Model temperature (0.0-2.0)
- **`llm_max_retries`**: Retry attempts on failure

## Error Handling

All LLM calls are wrapped in try-except blocks:
- **On failure**: Silently falls back to deterministic TF-IDF content
- **No exceptions propagated**: Ensures pipeline always completes
- **Budget validation**: LLM outputs are validated against config limits before use

## Design Philosophy

The pipeline follows a **graceful degradation** pattern:
1. **Best case**: LLM enhances titles and insights for better quality
2. **Fallback case**: Deterministic TF-IDF ensures consistent, acceptable output
3. **No single point of failure**: Pipeline works with or without LLM

This design ensures:
- ✅ **Reliability**: Always produces output
- ✅ **Consistency**: Deterministic fallback for testing
- ✅ **Quality**: LLM enhancement when available
- ✅ **Cost control**: Limited to top N clusters

