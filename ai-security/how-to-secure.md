
## 🛠️ Model Artifacts Storage
- Implements **AES-256 encryption** for model weights and parameters at rest.

    ```python
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    import os
    import pickle
    
    # Sample model weights (just for illustration)
    model_weights = {'layer1': [0.1, 0.2, 0.3], 'layer2': [0.4, 0.5, 0.6]}
    
    # AES-256 Encryption Setup
    key = os.urandom(32)  # 256-bit random key for AES-256
    iv = os.urandom(16)   # Initialization vector for the encryption
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Serialize the model weights into a byte stream
    model_weights_bytes = pickle.dumps(model_weights)
    
    # Ensure the data is a multiple of block size (AES block size = 16)
    padding_length = 16 - len(model_weights_bytes) % 16
    model_weights_bytes += b'\0' * padding_length  # Padding
    
    # Encrypt the model weights
    encrypted_model_weights = encryptor.update(model_weights_bytes) + encryptor.finalize()
    
    # Save the encrypted model weights to a file
    with open('encrypted_model_weights.bin', 'wb') as file:
        file.write(iv + encrypted_model_weights)  # Save the IV alongside the encrypted data
    
    print("Model weights encrypted and saved successfully!")
    ```


- Decrypting Model Weights
When the model is needed for predictions or further training, it can be decrypted using the corresponding key and IV. This ensures that the model weights remain secure during storage, while still being accessible when needed.

    ```python
    # Load the encrypted model weights from the file
    with open('encrypted_model_weights.bin', 'rb') as file:
        iv = file.read(16)  # The first 16 bytes are the IV
        encrypted_data = file.read()
    
    # Decrypt the model weights
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_model_weights_bytes = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # Unpickle the decrypted data
    decrypted_model_weights = pickle.loads(decrypted_model_weights_bytes)
    
    print("Decrypted model weights:", decrypted_model_weights)
    ```

## Uses HashiCorp Vault for secrets management.
- HashiCorp Vault is used to manage and store sensitive information, such as API keys, passwords, and other credentials required for securely accessing and interacting with model artifacts.
  ```python
  import hvac

  # Create a client to interact with HashiCorp Vault
  client = hvac.Client(url='http://127.0.0.1:8200', token='your-vault-token')
  
  # Store a secret
  client.secrets.kv.v2.create_or_update_secret(path='model/secret', secret={'api_key': 'your-api-key'})
  
  # Retrieve the secret
  secret = client.secrets.kv.v2.read_secret_version(path='model/secret')
  api_key = secret['data']['data']['api_key']
  
  print("API Key:", api_key)
  ```
## Example (storing and retrieving secrets):
  ```python
    import hvac
  
    # Create a client to interact with HashiCorp Vault
    client = hvac.Client(url='http://127.0.0.1:8200', token='your-vault-token')
    
    # Store a secret
    client.secrets.kv.v2.create_or_update_secret(path='model/secret', secret={'api_key': 'your-api-key'})
    
    # Retrieve the secret
    secret = client.secrets.kv.v2.read_secret_version(path='model/secret')
    api_key = secret['data']['data']['api_key']
    
    print("API Key:", api_key)
  ```
## Maintains Segregated Storage with RBAC for Different Environments
- Role-Based Access Control (RBAC) is implemented to ensure different environments (e.g., development, testing, production) are segregated, and only authorized users have access to specific model artifacts.

## Example (Configuring RBAC in Vault):
- In Vault, you can configure RBAC policies to control access to secrets based on roles.
- Example Vault policy for development environment:
```python
  path "model/development/*" {
    capabilities = ["create", "read", "update", "delete"]
  }
  
  path "model/production/*" {
    capabilities = ["deny"]
  }
```
  



  
