import os, threading
from functools import lru_cache
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

# Thread-safe lazy singleton for the worker process
_client = None
_lock = threading.Lock()

def get_client() -> CosmosClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = CosmosClient(
                    os.environ["AZURE_DB_URI"],                  # e.g. https://acct.documents.azure.com
                    credential=DefaultAzureCredential(),         # MI in Azure, dev identity locally
                    consistency_level="Session",
                )
    return _client

@lru_cache(maxsize=16)
def get_db(db_name: str):
    return get_client().get_database_client(db_name)

@lru_cache(maxsize=64)
def get_container(db_name: str, container_name: str):
    return get_db(db_name).get_container_client(container_name)