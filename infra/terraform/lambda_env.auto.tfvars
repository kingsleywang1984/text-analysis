# Lambda environment variables (AppConfig) loaded by src/config.py
#
# This repo has a `.env.example` (ignored by tooling in some environments). This tfvars file
# mirrors those defaults so Terraform deploys the same config as local dev/test.
#
# Terraform automatically loads `*.auto.tfvars` from the working directory, so running
# `terraform apply` inside `infra/terraform` will pick this up by default.

lambda_env = {
  # Embedding
  EMBEDDING_PROVIDER            = "tfidf"
  EMBEDDING_MODEL               = "tfidf"
  EMBEDDING_API_BASE_URL        = "https://api.openai.com"
  EMBEDDING_API_KEY             = ""
  EMBEDDING_API_TIMEOUT_SECONDS = "30"
  EMBEDDING_API_BATCH_SIZE      = "256"
  EMBEDDING_TFIDF_MAX_FEATURES  = "2048"
  EMBEDDING_TFIDF_NGRAM_MIN     = "1"
  EMBEDDING_TFIDF_NGRAM_MAX     = "2"

  # Clustering
  CLUSTER_SIMILARITY_THRESHOLD = "0.55"
  CLUSTER_MAX_CLUSTERS         = "10"
  CLUSTER_OVERFLOW_STRATEGY    = "OTHER"

  # Sentiment
  SENTIMENT_STRONG_NEGATIVE_THRESHOLD = "-0.5"
  SENTIMENT_POSITIVE_THRESHOLD        = "0.3"
  SENTIMENT_NEGATIVE_THRESHOLD        = "-0.2"

  # LLM output limits
  CLUSTER_INSIGHTS_MIN        = "2"
  CLUSTER_INSIGHTS_MAX        = "3"
  COMPARISON_SIMILARITIES_MIN = "1"
  COMPARISON_SIMILARITIES_MAX = "3"
  COMPARISON_DIFFERENCES_MIN  = "1"
  COMPARISON_DIFFERENCES_MAX  = "3"

  # LLM config
  # If you enable a provider (LLM_PROVIDER=openai_compatible), you must also set:
  # - LLM_BASE_URL
  # - LLM_API_KEY
  # - LLM_MODEL
  LLM_PROVIDER = "none"
  LLM_BASE_URL = "https://openrouter.ai/api"
  LLM_API_KEY  = "<place openrouter api key here>"
  LLM_MODEL    = "openai/gpt-oss-20b:free"

  # LLM runtime tuning (required by src/config.py regardless of provider)
  LLM_TIMEOUT_SECONDS = "8"
  LLM_TEMPERATURE     = "0.2"
  LLM_MAX_RETRIES     = "1"

  # LLM bounded controls (optional in src/config.py; defaults exist)
  LLM_MAX_CLUSTERS                         = "10"
  LLM_REPRESENTATIVE_SENTENCES_PER_CLUSTER = "10"
}
