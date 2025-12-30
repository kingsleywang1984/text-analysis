# Text Insight Clustering Service

A production-ready text analysis service that clusters user feedback sentences, analyzes sentiment, and generates insights using semantic similarity and optional LLM enhancement. Designed for deployment on AWS Lambda with API Gateway integration.

## Features

- **Semantic Clustering**: Groups similar sentences using TF-IDF or OpenAI embeddings with greedy threshold clustering
- **Sentiment Analysis**: VADER-based sentiment scoring at sentence and cluster levels
- **Two Analysis Modes**:
  - **Standalone**: Analyze a single set of feedback with titles and key insights
  - **Comparison**: Compare baseline vs. comparison cohorts with similarities and differences
- **LLM Enhancement** (Optional): Improves cluster titles and insights using OpenAI-compatible APIs
- **Graceful Degradation**: Falls back to deterministic TF-IDF when LLM is unavailable
- **Deterministic Output**: Reproducible results for testing and debugging

## Architecture

The service follows a pipeline architecture:

1. **Text Normalization**: Unicode normalization and whitespace cleanup
2. **Sentiment Analysis**: Sentence-level sentiment scoring using VADER
3. **Semantic Clustering**: Embedding-based similarity clustering
4. **Cluster Selection**: Top-N selection with overflow handling
5. **Report Generation**: Aggregated cluster reports with representative texts
6. **LLM Enhancement** (Optional): Title and insight generation
7. **Output Formatting**: Mode-specific response generation

For detailed LLM usage analysis, see [LLM_USAGE_ANALYSIS.md](./LLM_USAGE_ANALYSIS.md).

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd text-analysis

# Install dependencies
uv sync
```

### Configuration

The application loads configuration from environment variables (see `src/config.py`).

Recommended workflow:
- Copy `.env.example` to `.env`
- Edit `.env` values for your local environment

> Note: `*_BASE_URL` values should **NOT** include a trailing `/v1` — the app appends `/v1/...` itself.

```bash
# For the full, commented reference config, see `.env.example`.
# Below is the MINIMAL configuration required to pass `src/config.py` validation.
# All other variables are optional or have defaults; see `.env.example` for details.

# Required (always)
EMBEDDING_MODEL=tfidf
CLUSTER_SIMILARITY_THRESHOLD=0.55
CLUSTER_MAX_CLUSTERS=10
SENTIMENT_STRONG_NEGATIVE_THRESHOLD=-0.5
SENTIMENT_POSITIVE_THRESHOLD=0.3
SENTIMENT_NEGATIVE_THRESHOLD=-0.2

# Required budgets (always)
CLUSTER_INSIGHTS_MIN=2
CLUSTER_INSIGHTS_MAX=3
COMPARISON_SIMILARITIES_MIN=1
COMPARISON_SIMILARITIES_MAX=3
COMPARISON_DIFFERENCES_MIN=1
COMPARISON_DIFFERENCES_MAX=3

# LLM toggles (always required by config loader; provider can still be disabled)
LLM_PROVIDER=none
LLM_TIMEOUT_SECONDS=8
LLM_TEMPERATURE=0.2
LLM_MAX_RETRIES=1

# If you enable a real LLM (LLM_PROVIDER=openai_compatible), you must also set:
# LLM_BASE_URL=https://openrouter.ai/api
# LLM_API_KEY=your-key
# LLM_MODEL=openai/gpt-4o-mini

# If you enable OpenAI embeddings (EMBEDDING_PROVIDER=openai), you must also set:
# EMBEDDING_PROVIDER=openai
# EMBEDDING_API_BASE_URL=https://api.openai.com
# EMBEDDING_API_KEY=your-key
```

### Running Locally

#### Standalone Mode Example

```bash
python scripts/run_standalone_example.py
```

#### Comparison Mode Example

```bash
python scripts/run_comparison_example.py
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline_analyze_e2e.py

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## API Usage

### Request Format

**Endpoint**: `POST /analyze`

**Request Body**:

```json
{
  "surveyTitle": "Customer Feedback Survey",
  "theme": "payment issues",
  "baseline": [
    {
      "sentence": "Withholding my money",
      "id": "comment-uuid-1"
    },
    {
      "sentence": "I want my money back",
      "id": "comment-uuid-2"
    }
  ],
  "comparison": [
    {
      "sentence": "Refund process is slow",
      "id": "comment-uuid-3"
    }
  ],
  "query": "optional query string"
}
```

### Response Format

#### Standalone Mode

```json
{
  "clusters": [
    {
      "title": "Payment Withholding Complaints",
      "sentiment": "negative",
      "keyInsights": [
        "Users report significant delays in receiving refunds",
        "Payment processing issues are a primary concern"
      ]
    }
  ]
}
```

#### Comparison Mode

```json
{
  "clusters": [
    {
      "title": "Refund Processing Delays",
      "sentiment": "negative",
      "baselineSentences": ["comment-uuid-1", "comment-uuid-2"],
      "comparisonSentences": ["comment-uuid-3"],
      "keySimilarities": [
        "Both cohorts express frustration with payment delays"
      ],
      "keyDifferences": [
        "Baseline users emphasize withdrawal restrictions",
        "Comparison users focus on refund timelines"
      ]
    }
  ]
}
```

## Deployment

### Production Deployment

The service has been successfully deployed to AWS Lambda with API Gateway integration.

**API Gateway URL**: `https://pwh7rgqw92.execute-api.ap-southeast-2.amazonaws.com`

**Endpoint**: `POST /analyze`

#### Testing the Deployed Service

You can test the deployed service using the provided test scripts:

```bash
# Test standalone mode
python scripts/test_standalone_api.py

# Test comparison mode
python scripts/test_comparison_api.py
```

These scripts will send requests to the deployed API Gateway endpoint and display the response.

### AWS Lambda Deployment

The service is designed for AWS Lambda with API Gateway integration.

#### Build Docker Image

```bash
docker build -t text-analysis:latest .
```

#### Deploy with Terraform

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

See `infra/terraform/README.md` for detailed deployment instructions.

### Lambda Handler

The Lambda handler (`src.handler.lambda_handler`) supports:
- API Gateway REST API (v1) events
- API Gateway HTTP API (v2) events
- Base64-encoded request bodies

## Project Structure

```
text-analysis/
├── src/
│   ├── handler.py              # Lambda entrypoint
│   ├── config.py               # Configuration management
│   ├── models/
│   │   ├── schemas.py          # Request/response schemas
│   │   └── validators.py       # LLM output validators
│   ├── pipeline/
│   │   ├── analyze.py          # Main analysis pipeline
│   │   ├── pipeline.py         # Internal pipeline models
│   │   ├── clustering.py       # Clustering algorithm
│   │   ├── embedding.py         # Embedding providers
│   │   ├── sentiment.py        # Sentiment analysis
│   │   ├── aggregate.py        # Cluster aggregation
│   │   ├── insights_tfidf.py   # TF-IDF fallback generator
│   │   └── normalize.py        # Text normalization
│   └── llm/
│       ├── client.py           # LLM client implementations
│       ├── factory.py          # LLM client factory
│       └── prompts.py         # LLM prompt templates
├── tests/                      # Test suite
├── scripts/                    # Example scripts
├── data/                       # Sample input data
├── infra/                      # Infrastructure as code
│   └── terraform/             # Terraform configuration
└── Dockerfile                 # Lambda container image
```

## Key Design Decisions

### Combined Clustering for Comparison Mode

In comparison mode, baseline and comparison sentences are **combined and clustered together** (not clustered separately). This ensures:
- Cross-cohort semantic similarity discovery
- Comparable clusters (each cluster contains both baseline and comparison content)
- Natural alignment of similar themes across cohorts

### LLM-First Strategy with Fallback

The service uses an LLM-first approach with deterministic fallback:
- Tries LLM enhancement when available
- Falls back to TF-IDF if LLM fails or is unavailable
- Ensures the pipeline always produces output

### Deterministic Processing

All non-LLM steps are deterministic:
- Clustering results are reproducible
- Sentiment analysis uses VADER (deterministic)
- Text normalization is deterministic
- Enables reliable testing and debugging

## Configuration Reference

### Embedding Providers

- **TF-IDF**: Default, no external dependencies
- **OpenAI**: Requires `EMBEDDING_API_KEY` and `EMBEDDING_API_BASE_URL`

### Clustering Strategy

- **Similarity Threshold**: Minimum cosine similarity for cluster membership (0.0-1.0)
- **Max Clusters**: Maximum number of clusters to return
- **Overflow Strategy**:
  - `OTHER`: Merge overflow clusters into an "Other" cluster
  - `DROP`: Discard overflow clusters

### LLM Configuration

- **Provider**: `none` (disabled) or `openai_compatible`
- **Max Clusters**: Only top N clusters are enhanced with LLM
- **Representative Sentences**: Number of sentences sent to LLM per cluster

## Testing

The test suite includes:
- Unit tests for individual pipeline components
- Integration tests for end-to-end flows
- LLM schema validation tests
- Configuration loading tests

Run tests:
```bash
uv run pytest
```

## Development

### Adding New Features

1. Follow the existing pipeline architecture
2. Add tests for new functionality
3. Update configuration schema if needed
4. Document in relevant sections

### Code Style

- Type hints required
- Pydantic models for data validation
- Dataclasses for internal models
- Comprehensive docstrings

## License

[Add license information]

## Contributing

[Add contributing guidelines]

## Support

[Add support/contact information]

