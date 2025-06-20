import os
from typing import Dict, Any
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from azure.identity.aio import ChainedTokenCredential as AsyncChainedTokenCredential, ManagedIdentityCredential as AsyncManagedIdentityCredential, AzureCliCredential as AsyncAzureCliCredential
from azure.appconfiguration import AzureAppConfigurationClient
from azure.core.exceptions import AzureError

class AppConfigClient:

    credential = None
    aiocredential = None

    def __init__(self):
        """
        Bulk-loads all keys labeled 'orchestrator' and 'gpt-rag' into an in-memory dict,
        giving precedence to 'orchestrator' where a key exists in both.
        """
        
        client_id = os.getenv("AZURE_CLIENT_ID")
       
        endpoint = os.getenv("APP_CONFIG_ENDPOINT")
        if not endpoint:
            raise EnvironmentError("APP_CONFIG_ENDPOINT must be set")

        self.credential = ChainedTokenCredential(ManagedIdentityCredential(client_id=client_id), AzureCliCredential())
        self.aiocredential = AsyncChainedTokenCredential(AsyncManagedIdentityCredential(client_id=client_id), AsyncAzureCliCredential())
        client = AzureAppConfigurationClient(base_url=endpoint, credential=self.credential)

        self._settings: Dict[str, str] = {}

        # 1) Load everything labeled “orchestrator”
        try:
            for setting in client.list_configuration_settings(label_filter="orchestrator"):
                self._settings[setting.key] = setting.value
        except AzureError as e:
            raise RuntimeError(f"Failed to bulk-load 'orchestrator' settings: {e}")

        # 2) Load “gpt-rag” ones only if not already present
        try:
            for setting in client.list_configuration_settings(label_filter="gpt-rag"):
                self._settings.setdefault(setting.key, setting.value)
        except AzureError as e:
            raise RuntimeError(f"Failed to bulk-load 'gpt-rag' settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Returns the in-memory value for the given key.

        If the key was not found under either label, returns `default`.
        """
        return self._settings.get(key, default)
