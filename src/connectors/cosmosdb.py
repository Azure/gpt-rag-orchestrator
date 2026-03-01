import logging
from azure.cosmos.aio import CosmosClient
from dependencies import get_config

class CosmosDBClient:
    """
    CosmosDBClient with persistent connection pooling.
    A single CosmosClient is created once and reused across all operations,
    eliminating per-request TCP/TLS handshake overhead.
    """

    def __init__(self):
        """
        Initializes the Cosmos DB client with a persistent connection.
        """
        # ==== Load all config parameters in one place ====
        self.cfg = get_config()
        self.database_account_name = self.cfg.get("DATABASE_ACCOUNT_NAME")
        self.database_name = self.cfg.get("DATABASE_NAME")
        self.db_uri = f"https://{self.database_account_name}.documents.azure.com:443/"
        # ==== End config block ====

        # Persistent CosmosClient — reuses TCP connections across all operations
        self._client = CosmosClient(self.db_uri, credential=self.cfg.aiocredential)

    def _get_container(self, container_name):
        """Returns a container client from the persistent connection."""
        db = self._client.get_database_client(database=self.database_name)
        return db.get_container_client(container_name)

    async def list_documents(self, container_name) -> list:
        """
        Lists all documents from the given container.
        """
        container = self._get_container(container_name)

        # Correct usage without the outdated argument
        query = "SELECT * FROM c"
        items_iterable = container.query_items(query=query, partition_key=None)

        documents = []
        async for item in items_iterable:
            documents.append(item)

        return documents

    async def get_document(self, container, key) -> dict:
        container_client = self._get_container(container)
        try:
            document = await container_client.read_item(item=key, partition_key=key)
            logging.info(f"[cosmosdb] document {key} retrieved.")
        except Exception as e:
            document = None
            logging.info(f"[cosmosdb] document {key} does not exist.")
        return document

    async def create_document(self, container, key, body=None) -> dict:
        container_client = self._get_container(container)
        try:
            if body is None:
                body = {"id": key}
            else:
                body["id"] = key  # ensure the document id is set
            document = await container_client.create_item(body=body)
            logging.info(f"[cosmosdb] document {key} created.")
        except Exception as e:
            document = None
            logging.info(f"[cosmosdb] error creating document {key}. Error: {e}")
        return document

    async def update_document(self, container, document) -> dict:
        container_client = self._get_container(container)
        try:
            document = await container_client.replace_item(item=document["id"], body=document)
            logging.info(f"[cosmosdb] document updated.")
        except Exception as e:
            document = None
            logging.warning(f"[cosmosdb] could not update document: {e}", exc_info=True)
        return document


_cosmosdb_client_instance = None

def get_cosmosdb_client() -> CosmosDBClient:
    """Returns a singleton CosmosDBClient to reuse TCP connections."""
    global _cosmosdb_client_instance
    if _cosmosdb_client_instance is None:
        _cosmosdb_client_instance = CosmosDBClient()
    return _cosmosdb_client_instance
