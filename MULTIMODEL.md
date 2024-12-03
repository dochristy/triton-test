# Triton Server Setup and Model Deployment

This guide provides step-by-step instructions for setting up the Triton Inference Server, packaging a model, and deploying it.

---

## 1. Setting Up the Triton Docker Image

### Step 1.1: Pull the Triton Server Docker Image  
```bash
docker pull nvcr.io/nvidia/tritonserver:24.11-py3
```

### Step 1.2: Verify the Downloaded Docker Image
```
docker images
REPOSITORY                    TAG         IMAGE ID       CREATED       SIZE
nvcr.io/nvidia/tritonserver   24.11-py3   042cb8f39b     7 days ago    17.4GB
```
### Step 1.3: Tag the Docker Image for AWS ECR
```
docker tag nvcr.io/nvidia/tritonserver:24.11-py3 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3
```
### Step 1.4: Authenticate Docker with AWS ECR
```
aws ecr get-login-password --region us-east-1 --profile local | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
```
## Step 1.5: Push the Docker Image to AWS ECR
```
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/triton-repo:24.11-py3
```

## 2. Model Packaging and Deployment to S3

### Step 2.1: Prepare the Model Directory Structure
Ensure your model files are organized as follows:
```
./densenet_onnx/
├── densenet_labels.txt
├── config.pbtxt
└── 1/
    └── model.onnx
```

### Step 2.2: Package the Model Directory
Create a tarball of the model directory using the following command:
```
tar -czf densenet_model.tar.gz -C model_package .
```

### Step 2.3: Verify the Contents of the Tarball
```
tar tvf densenet_model.tar.gz
```
Expected output:

```
drwxr-xr-x  0 whatever admin       0 Dec  2 19:51 ./
drwxr-xr-x  0 whatever admin       0 Dec  2 20:01 ./densenet_onnx/
-rw-r--r--  0 whatever admin   10311 Dec  2 20:01 ./densenet_onnx/densenet_labels.txt
drwxr-xr-x  0 whatever admin       0 Dec  2 19:52 ./densenet_onnx/1/
-rw-r--r--  0 whatever admin     389 Dec  2 19:52 ./densenet_onnx/config.pbtxt
-rw-r--r--  0 whatever admin 32719461 Dec  2 19:52 ./densenet_onnx/1/model.onnx
```

### Step 2.4: Upload the Packaged Model to S3
```
aws s3 cp densenet_model.tar.gz s3://dry-bean-bucket-c/models/model.tar.gz --profile local
```

## 3. Create the Model in SageMaker

<img width="1332" alt="image" src="https://github.com/user-attachments/assets/a4d82e5c-80c0-469a-a819-17357932c515">

## 4. Create the Endpoint Configuration ( make sure the instance type is ml.g4dn.xlarge )

<img width="1378" alt="image" src="https://github.com/user-attachments/assets/827d6c79-85d1-4982-929c-2b2028823d0d">









