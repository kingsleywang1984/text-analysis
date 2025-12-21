## Terraform deployment (API Gateway HTTP API → Lambda container image)

This matches the design-doc shape (API Gateway → Lambda), using **Lambda container images** and **uv** for packaging.

### 1) Build & push the Lambda container image

- Build using the repo `Dockerfile` (uses `uv sync --frozen --no-dev`).
- Push the image to ECR.
- Capture the image URI: `lambda_image_uri`

### 2) Deploy with Terraform

From `personal/text-analysis/infra/terraform`:

```bash
terraform init
terraform apply \
  -var="aws_region=ap-southeast-2" \
  -var="name=text-insight-clustering-service" \
  -var="lambda_image_uri=<YOUR_ECR_IMAGE_URI>" \
  -var='lambda_env={EMBEDDING_MODEL="tfidf",CLUSTER_SIMILARITY_THRESHOLD="0.75",CLUSTER_MAX_CLUSTERS="10",CLUSTER_OVERFLOW_STRATEGY="OTHER",SENTIMENT_STRONG_NEGATIVE_THRESHOLD="-0.5",SENTIMENT_POSITIVE_THRESHOLD="0.3",SENTIMENT_NEGATIVE_THRESHOLD="-0.2",CLUSTER_INSIGHTS_MIN="2",CLUSTER_INSIGHTS_MAX="3",COMPARISON_SIMILARITIES_MIN="1",COMPARISON_SIMILARITIES_MAX="3",COMPARISON_DIFFERENCES_MIN="1",COMPARISON_DIFFERENCES_MAX="3",LLM_PROVIDER="none",LLM_TIMEOUT_SECONDS="8",LLM_TEMPERATURE="0.2",LLM_MAX_RETRIES="1",LLM_MAX_CLUSTERS="10",LLM_REPRESENTATIVE_SENTENCES_PER_CLUSTER="10"}'
```

Terraform will output:
- `api_base_url`
- `analyze_url` (POST)

### 3) Invoke

```bash
curl -X POST "$(terraform output -raw analyze_url)" \
  -H "Content-Type: application/json" \
  -d @../../data/input_example.json
```


