# NVIDIA Triton Server Deployment Guide for AWS SageMaker

This comprehensive guide walks through the process of setting up Triton Inference Server, packaging models, and deploying them on AWS SageMaker.

## Table of Contents
1. [Setting Up Triton Docker Image](#1-setting-up-triton-docker-image)
2. [Model Packaging and Deployment](#2-model-packaging-and-deployment-to-s3)
3. [SageMaker Configuration](#3-sagemaker-configuration)
4. [IAM Configuration](#4-iam-configuration)
5. [API Development](#5-api-development)
6. [Testing](#6-testing)

## 1. Setting Up Triton Docker Image

### Pull and Configure Docker Image
```bash
# Pull the Triton Server image
docker pull nvcr.io/nvidia/tritonserver:24.11-py3

# Verify the download
docker images

# Tag image for AWS ECR
docker tag nvcr.io/nvidia/tritonserver:24.11-py3 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3

# Authenticate with AWS ECR
aws ecr get-login-password --region us-east-1 --profile local | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3
```

## 2. Model Packaging and Deployment to S3

### Directory Structure
Organize your model files as follows:
```
./densenet_onnx/
├── densenet_labels.txt
├── config.pbtxt
└── 1/
    └── model.onnx
```

### Packaging Steps
```bash
# Create model package
tar -czf densenet_model.tar.gz -C model_package .

# Verify package contents
tar tvf densenet_model.tar.gz

# Upload to S3
aws s3 cp densenet_model.tar.gz s3://dry-bean-bucket-c/models/model.tar.gz --profile local
```

## 3. SageMaker Configuration

### Create SageMaker Resources
1. Create Model
2. Create Endpoint Configuration
   - Instance Type: ml.g4dn.xlarge
3. Create Endpoint

## 4. IAM Configuration

### Required IAM Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": [
                "arn:aws:s3:::dry-bean-bucket/*",
                "arn:aws:s3:::dry-bean-bucket/models/*"
            ]
        }
    ]
}
```

## 5. API Development

### Create Chalice Project
```bash
chalice new-project triton-app
```

### Configure Chalice
Create `.chalice/config.json`:
```json
{
  "version": "2.0",
  "app_name": "triton-app",
  "stages": {
    "dev": {
      "api_gateway_stage": "api",
      "environment_variables": {
        "SAGEMAKER_ENDPOINT_NAME": "triton-ep",
        "MODEL_BUCKET_NAME": "dry-bean-bucket"
      },
      "iam_role_arn": "arn:aws:iam::123456789:role/AmazonSageMakerServiceCatalogProductsExecutionRole"
    }
  }
}
```

### Create API Handler
Create `app.py`:
```python
from chalice import Chalice, BadRequestError
import boto3
import json

app = Chalice(app_name="triton-app")
sagemaker_client = boto3.client("sagemaker-runtime")

@app.route("/")
def index():
    return {"message": "SageMaker Multi-Model Endpoint API"}

@app.route("/predict", methods=["POST"])
def predict():
    request = app.current_request
    body = request.json_body

    if "model_name" not in body or "input_data" not in body:
        raise BadRequestError("Missing 'model_name' or 'input_data' in request body.")

    model_name = body["model_name"]
    input_data = body["input_data"]
    endpoint_name = "triton-ep"
    model_path = f"{model_name}"

    try:
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            TargetModel=model_path,
            Body=json.dumps(input_data),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return {"model_name": model_name, "result": result}
    except Exception as e:
        app.log.error(f"Error invoking SageMaker endpoint: {str(e)}")
        raise BadRequestError(f"Error invoking model: {str(e)}")
```

### Deploy API
```bash
chalice deploy --profile local
```

## 6. Testing 
( Refer: https://github.com/dochristy/triton-test/blob/main/README.md, for example, how to wget img1.jpg and pip install )

### Test Client Implementation
```python
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
import json

def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return preprocess(img).numpy()

# Prepare test data
transformed_img = rn50_preprocess()

# Create request payload
payload = {
    "model_name": "models/model.tar.gz",
    "input_data": {
        "inputs": [{
            "name": "data_0",
            "shape": transformed_img.shape,
            "datatype": "FP32",
            "data": transformed_img.tolist()
        }],
        "outputs": [{
            "name": "fc6_1",
            "parameters": {
                "class_count": "1000"
            }
        }]
    }
}

# Make API request
api_url = "https://vz3v5a3xp9.execute-api.us-east-1.amazonaws.com/api/predict"

try:
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    result = response.json()
    print("Prediction results:", result)
except requests.exceptions.RequestException as e:
    print(f"Error making request: {str(e)}")
    if hasattr(e.response, 'text'):
        print(f"Response content: {e.response.text}")
```

### Example Response
```json
{
    "model_name": "models/model.tar.gz",
    "result": {
        "model_name": "models/model.tar.gz",
        "model_version": "1",
        "outputs": [
            {
                "name": "fc6_1",
                "datatype": "FP32",
                "shape": [1000],
                "data": [0.20974071323871613, 2.377682685852051, ...]
            }
        ]
    }
}
```
