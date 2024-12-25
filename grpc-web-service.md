```
grpc-web-chalice
├── __pycache__
│   └── app.cpython-312.pyc
├── app.py
├── chalicelib
│   ├── __pycache__
│   │   └── greeting_pb2.cpython-312.pyc
│   ├── greeting.proto
│   └── greeting_pb2.py
├── chalicelib_1
├── requirements.txt
└── test.py
```


config.json
```json
{
  "version": "2.0",
  "app_name": "grpc-web-service",
  "stages": {
    "dev": {
      "api_gateway_stage": "api",
      "environment_variables": {}
    }
  }
}
```

greeting.proto
```proto
syntax = "proto3";

package greeting;

service Greeter {
    rpc SayHello (HelloRequest) returns (HelloResponse) {}
}

message HelloRequest {
    string name = 1;
}

message HelloResponse {
    string message = 1;
}
```
```python
python3 -m grpc_tools.protoc -I. --python_out=chalicelib chalicelib/greeting.proto
```

app.py
```python
from chalice import Chalice, ChaliceViewError
import logging

app = Chalice(app_name='grpc-web-service')
app.log.setLevel(logging.DEBUG)

# Optional: Add basic API key auth
@app.authorizer()
def api_key_auth(auth_request):
    if 'x-api-key' not in auth_request.token:
        raise Exception('Unauthorized')
    return auth_request.auth_response(['*'])

@app.route('/', methods=['GET'])
def index():
    return {'hello': 'world'}

@app.route('/greet', methods=['POST'])
def greet():
    try:
        app.log.debug("Request received")
        request_body = app.current_request.json_body
        name = request_body.get('name', '')
        return {'message': f"Hello, {name}!"}
    except Exception as e:
        app.log.error(f"Error occurred: {str(e)}")
        raise ChaliceViewError(str(e))

# Add an OPTIONS method for CORS
@app.route('/greet', methods=['OPTIONS'])
def greet_options():
    return {'message': 'OK'}
```
test.py
```python
import requests
import json
import sys
import time

def test_api(base_url, api_key=None):
    """Test the API with proper authentication"""
    headers = {
        'Content-Type': 'application/json'
    }

    # Add API key if provided
    if api_key:
        headers['x-api-key'] = 'weMt1YrCdpiVSHzbmUtx686Q1Av3TomapK91tKc'

    try:
        # Test the base endpoint
        print("\nTesting base endpoint...")
        response = requests.get(
            base_url,
            headers=headers
        )
        print(f"Base endpoint status: {response.status_code}")
        print(f"Response: {response.text}")

        # Test the greet endpoint
        print("\nTesting greet endpoint...")
        response = requests.post(
            f"{base_url}/greet",
            headers=headers,
            json={"name": "John"}
        )
        print(f"Greet endpoint status: {response.status_code}")
        print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else " https://wwj1c1ue38.execute-api.us-east-1.amazonaws.com/"
    api_key = sys.argv[2] if len(sys.argv) > 2 else None

    # Make sure URL ends with /api
    if not url.endswith('/api'):
        url = url.rstrip('/') + '/api'

    print(f"Testing API at: {url}")
    test_api(url, api_key)
```
```shell
chalice deploy --profile local
```

<img width="1771" alt="image" src="https://github.com/user-attachments/assets/1452c1e3-3f50-44d4-8506-609c68006ec6" />



Testing:
```shell
(base) grpc-web-chalice % python3 test.py
Testing API at:  https://wwj1c1ue38.execute-api.us-east-1.amazonaws.com/api

Testing base endpoint...
Base endpoint status: 200
Response: {"hello":"world"}

Testing greet endpoint...
Greet endpoint status: 200
Response: {"message":"Hello, John!"}
```
