# shared/blob_client_async.py
import os, asyncio
from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

_client: Optional[BlobServiceClient] = None
_lock = asyncio.Lock()

async def get_blob_service_client() -> BlobServiceClient:
    """Process-wide async BlobServiceClient singleton."""
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    _client = BlobServiceClient.from_connection_string(conn_str)
                else:
                    account = os.environ["AZURE_STORAGE_ACCOUNT"]
                    _client = BlobServiceClient(
                        account_url=f"https://{account}.blob.core.windows.net",
                        credential=DefaultAzureCredential(),
                    )
    return _client