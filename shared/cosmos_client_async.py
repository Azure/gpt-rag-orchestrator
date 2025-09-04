import os, threading
from functools import lru_cache
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

# Thread-safe lazy singleton for the worker process
_client = None
_lock = threading.Lock()

AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

def get_client() -> CosmosClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = CosmosClient(
                    AZURE_DB_URI,
                    credential=DefaultAzureCredential(),
                    consistency_level="Session",
                )
    return _client

@lru_cache(maxsize=16)
def get_db(db_name: str):
    return get_client().get_database_client(db_name)

@lru_cache(maxsize=64)
def get_container(db_name: str, container_name: str):
    return get_db(db_name).get_container_client(container_name)