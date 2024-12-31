# Triton Inference Server with AWS Lambda Integration

A complete solution for deploying NVIDIA Triton Inference Server on AWS EC2 with a serverless API interface using AWS Lambda and Chalice framework.

## Architecture

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'arial', 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart LR
    Client[Client]
    API[API Gateway]
    Lambda[AWS Lambda]
    EC2[EC2 t2.small]
    S3[(S3 Bucket)]
    ECR[(ECR Registry)]
    
    subgraph AWS Cloud
        API --> |API Key Auth|Lambda
        Lambda --> |gRPC :8001|EC2
        EC2 --> |Pull Models|S3
        EC2 --> |Pull Image|ECR
    end
    
    Client --> |HTTPS|API
    
    style Client fill:#f5f5f5,stroke:#000000
    style AWS fill:#f5f5f5,stroke:#000000
    style API fill:#f5f5f5,stroke:#000000
    style Lambda fill:#f5f5f5,stroke:#000000
    style EC2 fill:#f5f5f5,stroke:#000000
    style S3 fill:#f5f5f5,stroke:#000000
    style ECR fill:#f5f5f5,stroke:#000000
```

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Docker installed
- Python 3.x
- S3 bucket for model storage

## Project Structure

```
models/
└── densenet_onnx
    ├── 1
    │   ├── config.pbtxt
    │   └── model.onnx
    └── densenet_labels.txt

triton-grpc-app/
├── app.py
├── requirements.txt
└── .chalice/
    └── config.json
```

## Setup Instructions

### 1. Docker Image Setup

```bash
# Pull NVIDIA Triton Server image
docker pull nvcr.io/nvidia/tritonserver:24.11-py3

# Tag for ECR
docker tag nvcr.io/nvidia/tritonserver:24.11-py3 <account_id>.dkr.ecr.us-east-1.amazonaws.com/triton-grpc:24.11-py3

# Push to ECR
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/triton-grpc:24.11-py3
```

### 2. EC2 Configuration

1. Launch t2.small instance with 24GB EBS
2. Configure security group to allow inbound traffic on port 8001 (gRPC)
3. Run Triton server:

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    <acct_id>.dkr.ecr.us-east-1.amazonaws.com/triton-grpc:24.11-py3 \
    tritonserver --model-repository=s3://dry-bean-bucket-c/models
```

### 3. Chalice Application Setup

1. Create new project:
```bash
chalice new-project triton-grpc-app
cd triton-grpc-app
```

2. Create requirements.txt:
```
chalice
numpy
tritonclient[grpc]
```

3. Configure .chalice/config.json:
```json
{
  "version": "2.0",
  "app_name": "triton-grpc-app",
  "stages": {
    "dev": {
      "api_gateway_stage": "api"
    }
  }
}
```

4. Create app.py:
```python
from chalice import Chalice
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

app = Chalice(app_name='triton-grpc-app')

# Server configuration
TRITON_GRPC_URL = "ec2-xx-xx-xx-xx.compute-1.amazonaws.com:8001"  # Replace with your host
MODEL_NAME = "densenet_onnx"
INPUT_NAME = "data_0"
OUTPUT_NAME = "fc6_1"
INPUT_SHAPE = [1, 3, 224, 224]

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Initialize client
        triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

        # Prepare input data
        input_data = np.random.rand(*INPUT_SHAPE).astype(np.float32)
        inputs = [
            grpcclient.InferInput(INPUT_NAME, INPUT_SHAPE,
                                  np_to_triton_dtype(input_data.dtype))
        ]
        inputs[0].set_data_from_numpy(input_data)

        # Define outputs
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        # Make inference request
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )

        output_data = response.as_numpy(OUTPUT_NAME)

        return {
            'status': 'success',
            'output': output_data.tolist()
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/health', methods=['GET'])
def health():
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        if triton_client.is_server_live():
            return {'status': 'healthy'}
        return {'status': 'unhealthy'}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}
```

5. Deploy the application:
```bash
chalice deploy --profile local
```

## API Usage

### Health Check
```bash
curl https://gm9b8nx3d8.execute-api.us-east-1.amazonaws.com/api/health
```

Expected response:
```json
{"status": "healthy"}
```

### Inference
```bash
curl https://gm9b8nx3d8.execute-api.us-east-1.amazonaws.com/api/infer \
--header 'Content-Type: application/json' \
--header 'x-api-key: YOUR_API_KEY' \
--data '{
    "name": "Test inference"
}'
```

Example response:
```json
{
    "status": "success",
    "output": [[[-1.0031218528747559], [-0.9455380439758301], ...]]
}
```


# For Multi Model
```python
from chalice import Chalice, Response
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

app = Chalice(app_name='triton-grpc-app')

# Server configuration
TRITON_GRPC_URL = "ec2-xx-xx-xx-xx.compute-1.amazonaws.com:8001"

# Model configurations
MODEL_CONFIGS = {
    "densenet": {
        "name": "densenet_onnx",
        "input_name": "data_0",
        "output_name": "fc6_1",
        "input_shape": [1, 3, 224, 224]
    },
    "resnet50": {
        "name": "resnet50_onnx",
        "input_name": "data",
        "output_name": "resnetv24_dense0_fwd",
        "input_shape": [1, 3, 224, 224]
    },
    "chrisnet": {
        "name": "densenet_onnx",
        "input_name": "data_0",
        "output_name": "fc6_1",
        "input_shape": [1, 3, 224, 224]
    }
}

def preprocess_image(raw_input, shape):
    """
    Placeholder for image preprocessing.
    In real application, this would handle actual image data.
    """
    return np.random.rand(*shape).astype(np.float32)

@app.route('/infer/{model_type}', methods=['POST'])
def infer(model_type):
    if model_type not in MODEL_CONFIGS:
        return Response(
            body={'error': f'Model {model_type} not supported. Available models: {list(MODEL_CONFIGS.keys())}'},
            status_code=400
        )

    try:
        model_config = MODEL_CONFIGS[model_type]

        # Initialize client
        triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

        # Prepare input data
        input_data = preprocess_image(None, model_config["input_shape"])
        inputs = [
            grpcclient.InferInput(
                model_config["input_name"],
                model_config["input_shape"],
                np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)

        # Define outputs
        outputs = [grpcclient.InferRequestedOutput(model_config["output_name"])]

        # Make inference request
        response = triton_client.infer(
            model_name=model_config["name"],
            inputs=inputs,
            outputs=outputs
        )

        output_data = response.as_numpy(model_config["output_name"])

        return {
            'status': 'success',
            'model': model_type,
            'output': output_data.tolist()
        }

    except Exception as e:
        return Response(
            body={
                'status': 'error',
                'model': model_type,
                'message': str(e)
            },
            status_code=500
        )

@app.route('/models', methods=['GET'])
def list_models():
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        model_repository = triton_client.get_model_repository_index()
        return {
            'status': 'success',
            'models': [model.name for model in model_repository.models]
        }
    except Exception as e:
        return Response(
            body={
                'status': 'error',
                'message': str(e)
            },
            status_code=500
        )

@app.route('/health', methods=['GET'])
def health():
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        if triton_client.is_server_live():
            return {'status': 'healthy'}
        return {'status': 'unhealthy'}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}
```

S3 Structure
```
S3://YOUR_BUCKET/models

tree models
models
├── chrisnet_onnx
│   ├── 1
│   │   ├── config.pbtxt
│   │   └── model.onnx
│   └── densenet_labels.txt
├── densenet_onnx
│   ├── 1
│   │   ├── config.pbtxt
│   │   └── model.onnx
│   └── densenet_labels.txt
└── resnet50_onnx
    ├── 1
    │   ├── config.pbtxt
    │   └── model.onnx
    └── resnet50_labels.txt
```

## API Usage

### List Models
```bash
curl -X GET https://gm9b8nx3d7.execute-api.us-east-1.amazonaws.com/api/models \ 
--header 'x-api-key: YOUR_API_KEY'  
```

Expected response:
```json
{"status":"success","models":["chrisnet_onnx","densenet_onnx","resnet50_onnx"]}
```

### Inference [ densenet ]
```bash
curl https://gm9b8nx3d7.execute-api.us-east-1.amazonaws.com/api/infer/densenet \
--header 'Content-Type: application/json' \
--header 'x-api-key: YOUR_API_KEY' \
--data '{
    "name": "Test Job"
}'
```

Example response:
```json
{
    "status": "success",
    "output": [[[-1.0031218528747559], [-0.9455380439758301], ...]]
}
```

### Inference [ chrisnet ]

```bash
curl https://gm9b8nx3d7.execute-api.us-east-1.amazonaws.com/api/infer/chrisnet \
--header 'Content-Type: application/json' \
--header 'x-api-key: YOUR_API_KEY' \
--data '{
    "name": "Test Job"
}'
```

Example response:
```json
{
    "status": "success",
    "output": [[[-1.0031218528747559], [-0.9455380439758301], ...]]
}
```



## Security Features

- API Gateway authentication using API keys
- EC2 security group limiting access to gRPC port (8001)
- IAM roles controlling service access
- S3 bucket policies for model access
- ECR repository policies

## Performance Considerations

1. EC2 Instance:
   - Monitor resource usage
   - Consider auto-scaling for production
   - Adjust instance type based on load

2. Lambda Function:
   - Optimize memory allocation
   - Implement connection pooling
   - Consider caching responses

3. Model Serving:
   - Configure model concurrency
   - Monitor inference latency
   - Consider batch processing

## Monitoring

1. CloudWatch Integration:
   - Lambda function logs
   - EC2 metrics
   - API Gateway access logs

2. Health Checks:
   - Endpoint availability
   - Error rates
   - Response times

## Known Limitations

- Fixed input shape
- Single model support
- Basic error handling
- No auto-scaling configuration
- Limited request validation

## Future Improvements

1. Technical:
   - Multiple model version support
   - Input preprocessing
   - Enhanced validation
   - Better error handling

2. Operational:
   - Auto-scaling setup
   - Advanced monitoring
   - Response caching
   - Load balancing

3. Security:
   - VPC integration
   - Enhanced authentication
   - Request encryption

## Troubleshooting

Common issues and solutions:

1. Connection Timeouts:
   - Check EC2 security groups
   - Verify Lambda execution role
   - Test direct gRPC connection

2. Memory Errors:
   - Increase Lambda memory
   - Monitor EC2 resources
   - Check input data size

3. Model Loading Failures:
   - Verify S3 permissions
   - Check model repository structure
   - Validate model configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Add your chosen license]

## Support

- Open an issue in the repository
- Contact the maintainers
- Check documentation

---
*Last Updated: December 2024*
