import hvac
import os
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import asyncio
import aiohttp
from typing import Dict, Optional

class VaultCredentialManager:
    def __init__(self, vault_url: str, vault_token: str):
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.encryption_key = self._derive_encryption_key()
        
    def _derive_encryption_key(self) -> bytes:
        password = os.environ.get('QUANTUM_MASTER_KEY', 'default_key').encode()
        salt = b'quantum_trading_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
        
    async def store_credentials(self, path: str, credentials: Dict) -> bool:
        try:
            # Encrypt sensitive data
            fernet = Fernet(self.encryption_key)
            encrypted_creds = {}
            
            for key, value in credentials.items():
                if isinstance(value, str):
                    encrypted_value = fernet.encrypt(value.encode()).decode()
                    encrypted_creds[key] = encrypted_value
                else:
                    encrypted_creds[key] = value
                    
            # Store in Vault
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=encrypted_creds
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to store credentials: {e}")
            return False
            
    async def retrieve_credentials(self, path: str) -> Optional[Dict]:
        try:
            # Retrieve from Vault
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            encrypted_creds = response['data']['data']
            
            # Decrypt sensitive data
            fernet = Fernet(self.encryption_key)
            decrypted_creds = {}
            
            for key, value in encrypted_creds.items():
                if isinstance(value, str):
                    try:
                        decrypted_value = fernet.decrypt(value.encode()).decode()
                        decrypted_creds[key] = decrypted_value
                    except:
                        decrypted_creds[key] = value
                else:
                    decrypted_creds[key] = value
                    
            return decrypted_creds
            
        except Exception as e:
            print(f"Failed to retrieve credentials: {e}")
            return None