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
