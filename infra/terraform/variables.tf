variable "aws_region" {
  type        = string
  description = "AWS region to deploy into"
  default     = "ap-southeast-2"
}

variable "name" {
  type        = string
  description = "Base name for resources (Lambda, API, etc.)"
  default     = "text-insight-clustering-service"
}

variable "lambda_image_uri" {
  type        = string
  description = "ECR image URI for the Lambda container image (e.g. <account>.dkr.ecr.<region>.amazonaws.com/repo:tag)"
}

variable "lambda_timeout_seconds" {
  type        = number
  description = "Lambda timeout (seconds)"
  default     = 60
}

variable "lambda_memory_mb" {
  type        = number
  description = "Lambda memory (MB)"
  default     = 2048
}

variable "lambda_architectures" {
  type        = list(string)
  description = "Lambda architectures (e.g. [\"x86_64\"] or [\"arm64\"])"
  default     = ["arm64"]
}

variable "log_retention_days" {
  type        = number
  description = "CloudWatch Logs retention for Lambda"
  default     = 14
}

variable "lambda_env" {
  type        = map(string)
  description = "Environment variables passed to Lambda (same keys as .env.example)"
  default     = {}
}
