# providers.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS profile to use"
  type        = string
  default     = "local"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "triton"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ecr.tf
resource "aws_ecr_repository" "triton_repo" {
  name                 = "triton-repo"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# s3.tf
resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-model-bucket-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# iam.tf
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "${var.project_name}-sagemaker-s3-policy"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_bucket.arn,
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# sagemaker.tf
resource "aws_sagemaker_model" "triton_model" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "${aws_ecr_repository.triton_repo.repository_url}:24.11-py3"
    mode  = "MultiModel"
    environment = {
      SAGEMAKER_PROGRAM           = "serve"
      SAGEMAKER_SUBMIT_DIRECTORY  = "/opt/ml/model"
      SAGEMAKER_CONTAINER_LOG_LEVEL = "20"
      MAX_BATCH_SIZE              = "8"
      MODEL_S3_BUCKET             = aws_s3_bucket.model_bucket.id
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "triton_config" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "default"
    model_name            = aws_sagemaker_model.triton_model.name
    instance_type         = "ml.g4dn.xlarge"
    initial_instance_count = 1
  }
}

resource "aws_sagemaker_endpoint" "triton_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.triton_config.name
}

# api_gateway.tf
resource "aws_api_gateway_rest_api" "triton_api" {
  name        = "${var.project_name}-api"
  description = "API Gateway for Triton inference"
}

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.triton_api.id
  parent_id   = aws_api_gateway_rest_api.triton_api.root_resource_id
  path_part   = "predict"
}

resource "aws_api_gateway_method" "predict_post" {
  rest_api_id   = aws_api_gateway_rest_api.triton_api.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.triton_api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.predict_lambda.invoke_arn
}

# lambda.tf
resource "aws_lambda_function" "predict_lambda" {
  filename         = "lambda_function.zip"
  function_name    = "${var.project_name}-predict-lambda"
  role            = aws_iam_role.lambda_role.arn
  handler         = "app.predict"
  runtime         = "python3.9"

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = aws_sagemaker_endpoint.triton_endpoint.name
      MODEL_BUCKET_NAME       = aws_s3_bucket.model_bucket.id
    }
  }
}

resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_sagemaker_policy" {
  name = "${var.project_name}-lambda-sagemaker-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = [
          aws_sagemaker_endpoint.triton_endpoint.arn
        ]
      }
    ]
  })
}

# outputs.tf
output "ecr_repository_url" {
  value = aws_ecr_repository.triton_repo.repository_url
}

output "model_bucket_name" {
  value = aws_s3_bucket.model_bucket.id
}

output "sagemaker_endpoint_name" {
  value = aws_sagemaker_endpoint.triton_endpoint.name
}

output "api_gateway_url" {
  value = "${aws_api_gateway_rest_api.triton_api.execution_arn}/*/POST/predict"
}
