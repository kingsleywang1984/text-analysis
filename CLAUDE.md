# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-ready text analysis service that clusters user feedback sentences, analyzes sentiment, and generates insights using semantic similarity and optional LLM enhancement. Deployed as an AWS Lambda function with API Gateway integration.

## Common Commands

### Development

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline_analyze_e2e.py

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run standalone mode example locally
python scripts/run_standalone_example.py

# Run comparison mode example locally
python scripts/run_comparison_example.py

# Test deployed API (standalone)
python scripts/test_standalone_api.py

# Test deployed API (comparison)
python scripts/test_comparison_api.py
```

### Code Quality

```bash
# Format imports
uv run isort src tests scripts

# Lint with ruff
uv run ruff check src tests scripts
```

### Deployment

```bash
# Build Docker image for Lambda
docker build -t text-analysis:latest .

# Deploy with Terraform (from infra/terraform/)
terraform init
terraform plan
terraform apply
```

## Configuration

The application loads ALL configuration from environment variables via `src/config.py`. This is the **single source of truth** for configuration.

### Critical Configuration Rules

1. **Always use `.env` file for local development**: Copy `.env.example` to `.env` and modify values
2. **Base URLs must NOT include trailing `/v1`**: The application appends `/v1/...` paths automatically
3. **All config values are validated at startup**: Missing or invalid values cause immediate failure
4. **Config sync across sources**: When adding/modifying config:
   - Update `src/config.py` (validation logic)
   - Update `.env.example` (documentation and defaults)
   - Update `infra/terraform/lambda_env.auto.tfvars` (production values)
   - Update any relevant documentation

### Minimal Required Configuration

See `.env.example` for the complete reference. The minimal set required to start:

```bash
EMBEDDING_MODEL=tfidf
CLUSTER_SIMILARITY_THRESHOLD=0.55
CLUSTER_MAX_CLUSTERS=10
SENTIMENT_STRONG_NEGATIVE_THRESHOLD=-0.5
SENTIMENT_POSITIVE_THRESHOLD=0.3
SENTIMENT_NEGATIVE_THRESHOLD=-0.2
CLUSTER_INSIGHTS_MIN=2
CLUSTER_INSIGHTS_MAX=3
COMPARISON_SIMILARITIES_MIN=1
COMPARISON_SIMILARITIES_MAX=3
COMPARISON_DIFFERENCES_MIN=1
COMPARISON_DIFFERENCES_MAX=3
LLM_PROVIDER=none
LLM_TIMEOUT_SECONDS=8
LLM_TEMPERATURE=0.2
LLM_MAX_RETRIES=1
```

## Architecture

### Pipeline Flow

The service follows a deterministic pipeline architecture (`src/pipeline/analyze.py`):

1. **Text Normalization** (`normalize.py`): Unicode normalization (NFKC), whitespace cleanup
2. **Sentiment Analysis** (`sentiment.py`): VADER-based sentence-level sentiment scoring
3. **Semantic Clustering** (`clustering.py`): Greedy threshold-based clustering using embeddings
4. **Cluster Selection** (`analyze.py`): Top-N selection with overflow handling (OTHER or DROP strategy)
5. **Report Generation** (`aggregate.py`): Aggregated cluster reports with representative texts
6. **LLM Enhancement** (Optional, `llm/client.py`): Title and insight generation
7. **Output Formatting** (`models/schemas.py`): Mode-specific response generation

### Two Analysis Modes

**Standalone Mode**: Analyze a single dataset
- Input: `baseline` sentences only
- Output: Clusters with `title`, `sentiment`, `keyInsights`

**Comparison Mode**: Compare two datasets
- Input: `baseline` AND `comparison` sentences
- Output: Clusters with `title`, `sentiment`, `baselineSentences`, `comparisonSentences`, `keySimilarities`, `keyDifferences`
- **Critical**: Baseline and comparison sentences are **combined and clustered together** (not clustered separately), ensuring cross-cohort semantic similarity discovery

### LLM-First Strategy with Fallback

The service uses an **LLM-first approach** with deterministic fallback (`src/pipeline/insights_tfidf.py`):
- Attempts LLM enhancement when available and `LLM_PROVIDER != none`
- Falls back to TF-IDF-based deterministic generation if LLM fails or is unavailable
- **Circuit breaker**: After first LLM failure in a request, LLM is disabled for remaining clusters in that request
- Ensures the pipeline always produces output

### Embedding Providers

Two embedding options (`src/pipeline/embedding.py`):
- **TF-IDF** (default): Deterministic, no external dependencies, configured via `EMBEDDING_TFIDF_*` variables
- **OpenAI**: Requires `EMBEDDING_API_KEY` and `EMBEDDING_API_BASE_URL`

Set via `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` environment variables.

### LLM Client Architecture

LLM integration follows a **Protocol-based design** for vendor flexibility:

1. **Protocol Interface** (`src/llm/client.py:LLMClient`): Stable contract that pipeline depends on
2. **Implementations**:
   - `FakeLLMClient`: Deterministic fake for tests (no network)
   - `OpenAICompatibleClient`: Real HTTP client using OpenAI-compatible API
3. **Factory** (`src/llm/factory.py`): Creates appropriate client based on `config.llm_provider`
4. **Prompts** (`src/llm/prompts.py`): Centralized prompt templates

When adding new LLM providers, implement the `LLMClient` protocol and update the factory.

### Lambda Handler

The Lambda handler (`src/handler.py`) supports:
- API Gateway REST API (v1) events
- API Gateway HTTP API (v2) events
- Base64-encoded request bodies
- Structured JSON logging

Entry point: `src.handler.lambda_handler`

## Key Design Decisions

### Deterministic Processing

All **non-LLM steps are deterministic** for reliable testing and debugging:
- Clustering uses stable sorting and greedy selection
- Sentiment analysis uses VADER (deterministic)
- Text normalization is deterministic
- Overflow handling is deterministic

### Overflow Strategy

When clusters exceed `CLUSTER_MAX_CLUSTERS`:
- **OTHER** (default): Top (N-1) clusters + one merged "Other" cluster containing all overflow
- **DROP**: Top N clusters only, discard overflow

### Pydantic-Based Validation

- **Request/Response schemas** (`src/models/schemas.py`): Strict Pydantic models with `extra="forbid"`
- **LLM output validators** (`src/models/validators.py`): Validate LLM responses meet budget constraints
- **Config validation** (`src/config.py`): All environment variables validated at startup

### Logging

Structured JSON logging via `src/logging_utils.py`:
- `log_info(event_name, **context)`: Info-level events
- `log_warning(event_name, **context)`: Warning-level events
- `log_error(event_name, **context)`: Error-level events

All logs include event names for easy filtering and correlation.

## Testing

The test suite (`tests/`) includes:
- **Unit tests**: Individual pipeline components (clustering, sentiment, embedding, etc.)
- **Integration tests**: End-to-end flows (`test_pipeline_analyze_e2e.py`)
- **LLM schema validation**: Validate LLM output formats (`test_llm_schemas.py`)
- **Config validation**: Environment variable loading (`test_config.py`)
- **Handler tests**: Lambda event parsing and routing (`test_handler.py`)

All tests are deterministic and use fixtures from `tests/conftest.py`.

## Working with the Codebase

### Adding a New Configuration Variable

1. Add validation logic to `src/config.py:EnvConfigLoader.load()`
2. Add field to `AppConfig` dataclass
3. Document in `.env.example` with purpose, type, constraints, defaults
4. Update `infra/terraform/lambda_env.auto.tfvars` with production value
5. Add test coverage in `tests/test_config.py`

### Modifying the Pipeline

1. Follow the existing step-by-step structure in `src/pipeline/analyze.py`
2. Ensure new steps are deterministic (or have deterministic fallback)
3. Add comprehensive docstrings explaining input/output formats
4. Update tests for both unit and integration coverage
5. Consider overflow handling implications

### Adding a New LLM Provider

1. Implement `LLMClient` protocol in `src/llm/client.py`
2. Update `src/llm/factory.py:create_llm_client()` to support new provider
3. Add new provider value to `LLM_PROVIDER` validation in `src/config.py`
4. Add provider-specific config variables if needed
5. Add test coverage using `FakeLLMClient` pattern

### Deployment Changes

When modifying deployment configuration:
1. Update Terraform variables in `infra/terraform/variables.tf`
2. Update `infra/terraform/lambda_env.auto.tfvars` for environment variables
3. Update `infra/terraform/README.md` with new deployment steps
4. Test deployment in a staging environment first

## API Contract

### Request Format

```json
{
  "surveyTitle": "Customer Feedback Survey",
  "theme": "payment issues",
  "baseline": [{"sentence": "...", "id": "comment-uuid-1"}],
  "comparison": [{"sentence": "...", "id": "comment-uuid-2"}],  // optional
  "query": "optional query string"  // optional
}
```

### Response Format (Standalone)

```json
{
  "clusters": [
    {
      "title": "Payment Withholding Complaints",
      "sentiment": "negative",
      "keyInsights": ["...", "..."]
    }
  ]
}
```

### Response Format (Comparison)

```json
{
  "clusters": [
    {
      "title": "Refund Processing Delays",
      "sentiment": "negative",
      "baselineSentences": ["uuid-1", "uuid-2"],
      "comparisonSentences": ["uuid-3"],
      "keySimilarities": ["..."],
      "keyDifferences": ["..."]
    }
  ]
}
```

## Production Deployment

**Deployed API Gateway URL**: `https://pwh7rgqw92.execute-api.ap-southeast-2.amazonaws.com`

**Endpoint**: `POST /analyze`

See `infra/terraform/README.md` for full deployment instructions.
