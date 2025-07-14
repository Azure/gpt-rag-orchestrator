import os
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import AzureError
from appconfig import AppConfigClient

class KeyVaultClient:
    """
    Simple wrapper to fetch secrets from Azure Key Vault.
    Expects an environment variable KEY_VAULT_URI = "https://<your-vault-name>.vault.azure.net/"
    """
    def __init__(self):
        cfg = AppConfigClient()
        vault_uri = cfg.get("KEY_VAULT_URI")  
        if not vault_uri:
            raise EnvironmentError("KEY_VAULT_URI must be set to your Key Vault URI in App Configuration")
        credential = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
        try:
            self._client = SecretClient(vault_url=vault_uri, credential=credential)
        except AzureError as e:
            raise RuntimeError(f"Failed to create SecretClient: {e}")

    def get_secret(self, name: str) -> str:
        """
        Returns the secret value, or None if not found.
        """
        try:
            secret = self._client.get_secret(name)
            return secret.value
        except AzureError as e:
            # You might want to log or handle not found differently
            raise RuntimeError(f"Error retrieving secret '{name}': {e}")
