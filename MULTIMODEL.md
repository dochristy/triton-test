# Triton Server Setup and Model Deployment

This guide provides step-by-step instructions for setting up the Triton Inference Server, packaging a model, and deploying it.

---

## 1. Setting Up the Triton Docker Image

### Step 1.1: Pull the Triton Server Docker Image  
```bash
docker pull nvcr.io/nvidia/tritonserver:24.11-py3

### Step 1.2: Verify the Downloaded Docker Image
docker images
REPOSITORY                    TAG         IMAGE ID       CREATED       SIZE
nvcr.io/nvidia/tritonserver   24.11-py3   042cb8f39b     7 days ago    17.4GB

### Step 1.3: Tag the Docker Image for AWS ECR
docker tag nvcr.io/nvidia/tritonserver:24.11-py3 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3

### Step 1.4: Authenticate Docker with AWS ECR
aws ecr get-login-password --region us-east-1 --profile local | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

## Step 1.5: Push the Docker Image to AWS ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3

```
## 2. Model Packaging and Deployment to S3

### Step 2.1: Prepare the Model Directory Structure
Ensure your model files are organized as follows:

./densenet_onnx/
├── densenet_labels.txt
├── config.pbtxt
└── 1/
    └── model.onnx



```
