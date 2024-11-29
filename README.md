# Triton Inference Server DenseNet Demo

This repository demonstrates how to deploy and serve a DenseNet model using NVIDIA's Triton Inference Server. The demo includes a simple image classification example using a pre-trained DenseNet model in ONNX format.

## Prerequisites

- Docker
- Python 3.x
- NVIDIA GPU (optional)

## Project Structure

```
.
├── models
│   ├── densenet_onnx
│   │   ├── 1
│   │   │   └── model.onnx
│   │   ├── config.pbtxt
│   │   └── densenet_labels.txt
│   └── model1
│       ├── 1
│       │   └── model.onnx
│       └── config.pbtxt
├── client.py
└── start_triton_server.sh
```

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```
2. To get the densenet model
```chatinput
mkdir -p models/densenet_onnx/1
wget -O models/densenet_onnx/1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx

```

3. Create a DenseNet Labels file (densenet_labels.txt)
```
TENCH
GOLDFISH
WHITE SHARK
TIGER SHARK
HAMMERHEAD SHARK
ELECTRIC RAY
STINGRAY
ROOSTER
...
...
```


4. Start the Triton Server:
```bash
./start_triton_server.sh
```

5. In a new terminal, run the client container:
```bash
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.11-py3-sdk bash
```

6. Inside the client container, install required packages:
```bash
pip install torchvision tritonclient[all] gevent
```

7. Download a test image:
```bash
wget -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

## Model Configuration

The repository includes two model configurations:

### DenseNet ONNX Model
- Location: `models/densenet_onnx/`
- Input: 
  - Name: "data_0"
  - Shape: [1, 3, 224, 224]
  - Format: NCHW
  - Data Type: FP32
- Output:
  - Name: "fc6_1"
  - Shape: [1, 1000, 1, 1]
  - Labels file included

### Model1 (Additional Example)
- Location: `models/model1/`
- CPU optimization with OpenVINO
- Supports batching (max_batch_size: 8)

## Client Code

The `client.py` script demonstrates how to:
1. Preprocess an image using torchvision transforms
2. Create a Triton client connection
3. Send inference requests
4. Process the model's output

Key components:
```python
# Example preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Client setup and inference
client = httpclient.InferenceServerClient(url="localhost:8000")
```

client.py
```python
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype


# preprocessing function
def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Remove np.expand_dims() - just return the preprocessed tensor as numpy array
    return preprocess(img).numpy()


transformed_img = rn50_preprocess()

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

# Querying the server
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy("fc6_1").astype(str)

print(np.squeeze(inference_output)[:5])
```


## Running the Demo

1. Start the Triton server:
```bash
./start_triton_server.sh
```

2. In a separate terminal, run the client:
```bash
python3 client.py
```

Expected output format:
```
['11.548585:92:BEE EATER' '11.231406:14:INDIGO FINCH'
 '7.527274:95:JACAMAR' '6.922708:17:JAY' '6.576275:88:MACAW']
```
The output shows the top 5 predictions in the format "confidence_score:class_index".

## Server Configuration

The Triton server is configured with the following ports:
- 8000: HTTP interface
- 8001: gRPC interface
- 8002: Metrics interface

The server is started with the following command:
```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:24.11-py3 \
    tritonserver --model-repository=/models
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- NVIDIA Triton Inference Server
- DenseNet model architecture
# triton-test
