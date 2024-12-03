# Triton Server Setup and Multi Model Deployment in AWS Sagemaker

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

## 5. Create the Endpoint

<img width="1414" alt="image" src="https://github.com/user-attachments/assets/4882fef8-311d-4d49-832d-28ffbd44011a">

## 6. IAM Role/Policies

<img width="1301" alt="image" src="https://github.com/user-attachments/assets/f92724da-2a9f-4e7f-b1d1-fcfadb14355b">

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": "s3:GetObject",
			"Resource": [
				"arn:aws:s3:::dry-bean-bucket/*",
				"arn:aws:s3:::dry-bean-bucket/densenet_onnx/*"
			]
		}
	]
}
```

## 7. Chalice Project Creation
```
 chalice new-project triton-app
```

 <img width="265" alt="image" src="https://github.com/user-attachments/assets/c988e000-7dbb-4802-ba7e-632f22aa18a3">


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

### 8. app.py
```python
from chalice import Chalice, BadRequestError
import boto3
import json

app = Chalice(app_name="triton-app")

# Initialize SageMaker runtime client
sagemaker_client = boto3.client("sagemaker-runtime")


@app.route("/")
def index():
    return {"message": "SageMaker Multi-Model Endpoint API"}


@app.route("/predict", methods=["POST"])
def predict():
    request = app.current_request
    body = request.json_body

    # Validate the input
    if "model_name" not in body or "input_data" not in body:
        raise BadRequestError("Missing 'model_name' or 'input_data' in request body.")

    model_name = body["model_name"]
    input_data = body["input_data"]

    # Specify the SageMaker endpoint
    endpoint_name = "triton-ep"  # Replace with your SageMaker endpoint name

    model_path = f"{model_name}"

    try:
        # Call the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            TargetModel=model_path,
            Body=json.dumps(input_data),
        )

        # Parse the response
        result = json.loads(response["Body"].read().decode("utf-8"))
        return {"model_name": model_name, "result": result}

    except Exception as e:
        app.log.error(f"Error invoking SageMaker endpoint: {str(e)}")
        raise BadRequestError(f"Error invoking model: {str(e)}")

```

## 9. chalice deploy --profile local
```bash
    The LAMBDA will be created and also the endpoint
    (.venv) (base) MacBook-Pro triton-app % chalice deploy --profile local     
    Creating deployment package.
    Creating IAM role: triton-app-dev
    Creating lambda function: triton-app-dev
    Creating Rest API
    Resources deployed:
      - Lambda ARN: arn:aws:lambda:us-east-1:123456789:function:triton-app-dev
      - Rest API URL: https://vz3v5a3xp9.execute-api.us-east-1.amazonaws.com/api/


```

<img width="822" alt="image" src="https://github.com/user-attachments/assets/0bef6172-f1fd-45ef-939f-3f91aa8acbaf">

## 10. To Test
```python

root@docker-desktop:/opt/tritonserver# cat tclient.py
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).numpy()

# Preprocess the image
transformed_img = rn50_preprocess()

# Prepare the request payload
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

# API Gateway endpoint
api_url = "https://vz3v5a3xp9.execute-api.us-east-1.amazonaws.com/api/predict"

# Make the request
try:
    response = requests.post(api_url, json=payload)
    response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
    result = response.json()
    print("Prediction results:", result)

except requests.exceptions.RequestException as e:
    print(f"Error making request: {str(e)}")
    if hasattr(e.response, 'text'):
        print(f"Response content: {e.response.text}")
```

### OUTPUT:
root@docker-desktop:/opt/tritonserver# python3 tclient.py


Prediction results:

```
{'model_name': 'models/model.tar.gz', 'result':
{'model_name': 'models/model.tar.gz', 'model_version': '1',
'outputs':
[{'name': 'fc6_1', 'datatype': 'FP32', 'shape': [1000],
'data': [0.20974071323871613, 2.377682685852051, 1.0179954767227173, -0.6749277710914612,
0.9581587314605713, 1.922627568244934, 1.1472487449645996, 1.6970891952514648,
-0.000913398340344429, -1.3191059827804565, 6.046161651611328, 6.213389873504639,
5.619243621826172, 2.581407070159912, 11.231402397155762, 4.368611812591553, 3.8038957118988037,
6.922706604003906, 5.21445894241333, 4.204921722412109, 4.116398334503174, 4.906204700469971,
0.6235623359680176, 1.7377761602401733, 1.1507880687713623, -0.921630859375, -0.2747982144355774,
-0.7658523321151733, -2.178382396697998, -2.5586047172546387, 1.8045556545257568, 0.6859996914863586,
0.5720824003219604, -0.16455429792404175, 0.16581863164901733, -1.1402806043624878, 1.0850653648376465,
-0.7723713517189026, -0.6653391122817993, -1.679345726966858, 1.5801708698272705, -0.2765064239501953,
2.901796340942383, 0.42975834012031555, -0.8889715075492859, -1.4189574718475342, 1.9019747972488403,
2.4407997131347656, -0.9584187269210815, -0.444856733083725, -0.874829113483429,
-1.7724707126617432, -1.6850093603134155,...]}]}}
```












