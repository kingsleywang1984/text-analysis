terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }

  backend "s3" {
    bucket         = "terraform-state-103138678452-ap-southeast-2"
    key            = "text-insight-clustering-service/terraform.tfstate"
    region         = "ap-southeast-2"
    encrypt        = true
    # Use DynamoDB table for state locking (prevents concurrent modifications)
    dynamodb_table = "terraform-state-lock"
    # profile        = "kingsley-personal"
  }
}


