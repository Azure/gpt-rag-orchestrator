import os
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


def get_blob_service_client():
    """
    Create and return a BlobServiceClient instance.
    
    Returns:
        BlobServiceClient: Authenticated blob service client
    """
    storage_account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    if not storage_account_url:
        raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable not set")
    
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=storage_account_url, credential=credential)

