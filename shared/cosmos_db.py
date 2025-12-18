import os
from zoneinfo import ZoneInfo
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import uuid
import logging
from datetime import datetime, timezone


class CosmosDBClient:
    """Manages Azure Cosmos DB connections and operations."""

    def __init__(
        self,
        db_id: str = None,
        db_name: str = None,
        db_uri: str = None,
        consistency_level: str = "Session"
    ):
        self.db_id = db_id or os.environ.get("AZURE_DB_ID")
        self.db_name = db_name or os.environ.get("AZURE_DB_NAME")
        self.db_uri = db_uri or f"https://{self.db_id}.documents.azure.com:443/"
        self.consistency_level = consistency_level

        # Lazy init
        self._credential = None
        self._client = None
        self._db = None

    def _ensure_connected(self):
        """Lazy init"""
        if self._client is None:
            self._credential = DefaultAzureCredential()
            self._client = CosmosClient(
                self.db_uri,
                credential=self._credential,
                consistency_level=self.consistency_level
            )
            self._db = self._client.get_database_client(database=self.db_name)

    def store_agent_error(self, user_id: str, error: str, ask: str) -> None:
        try:
            self._ensure_connected()
            container = self._db.get_container_client("agentErrors")
            container.create_item(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "ask": ask,
                    "error": error,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as e:
            logging.error(f"[CosmosDB] Error storing agent error: {e}", exc_info=True)

    def get_conversation_data(
        self,
        conversation_id: str,
        user_id: str,
        type: str = None,
        user_timezone: str = None
    ) -> dict:
        self._ensure_connected()
        container = self._db.get_container_client("conversations")

        tz = timezone.utc
        if user_timezone:
            try:
                tz = ZoneInfo(user_timezone)
            except Exception:
                logging.warning(f"[CosmosDB] Unknown timezone '{user_timezone}', defaulting to UTC")

        try:
            conversation = container.read_item(
                item=conversation_id, partition_key=user_id
            )
        except Exception as e:
            logging.info(
                f"[CosmosDB] Conversation not found, creating new conversation_id: {conversation_id}"
            )
            conversation = container.create_item(
                body={
                    "id": conversation_id,
                    "user_id": user_id,
                    "conversation_data": {
                        "start_date": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
                        "history": [],
                        "memory_data": "",
                        "interaction": {},
                        "type": type if type else "default",
                    },
                }
            )

        conversation_data = conversation.get(
            "conversation_data",
            {
                "start_date": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
                "history": [],
                "memory_data": "",
                "interaction": {},
                "type": type if type else "default",
            },
        )

        # Ensure history exists
        if "history" not in conversation_data:
            logging.info(
                f"[CosmosDB] 'history' not found for conversation_id: {conversation_id}. Initializing."
            )
            conversation_data["history"] = []

            try:
                container.upsert_item(
                    {
                        "id": conversation_id,
                        "user_id": user_id,
                        "conversation_data": conversation_data,
                    }
                )
                logging.info(
                    f"[CosmosDB] 'history' initialized and updated for conversation_id: {conversation_id}."
                )
            except Exception as e:
                logging.error(
                    f"[CosmosDB] Failed to update 'history' for conversation_id: {conversation_id}. Error: {e}",
                    exc_info=True
                )

        return conversation_data

    def update_conversation_data(
        self,
        conversation_id: str,
        user_id: str,
        conversation_data: dict
    ) -> dict:
        try:
            self._ensure_connected()
            container = self._db.get_container_client("conversations")
            container.upsert_item(
                {
                    "id": conversation_id,
                    "user_id": user_id,
                    "conversation_data": conversation_data,
                }
            )
        except Exception as e:
            logging.error(f"[CosmosDB] Error updating the conversations: {e}", exc_info=True)

        return conversation_data

    def get_credit_table(self, container_name: str = "subscriptionsTiers") -> list:
        """
        Retrieve credit table/subscription tiers.
        """
        try:
            self._ensure_connected()
            container = self._db.get_container_client(container=container_name)
            credit_table = list(container.query_items(
                query="SELECT * FROM c WHERE c.id = 'requestCost'",
                enable_cross_partition_query=True
            ))
            return credit_table
        except Exception as e:
            logging.error(f"[CosmosDB] Error retrieving the credit table: {e}", exc_info=True)
            return None

    def update_user_credit(
        self,
        organization_id: str,
        user_id: str,
        credit_consumed: float,
        container_name: str = "organizationsUsage"
    ) -> dict:
        """
        Atomically increment currentUsed credit for a user.
        """
        try:
            self._ensure_connected()
            container = self._db.get_container_client(container=container_name)

            # Query for the org config document to find user index
            query = "SELECT * FROM c WHERE c.organizationId = @organization_id"
            parameters = [{"name": "@organization_id", "value": organization_id}]
            results = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if not results:
                logging.error(f"[CosmosDB] Organization config not found for organization_id: {organization_id}")
                return None

            org_config = results[0]

            # Find the user's index in the allowedUserIds array (should've been validated in the FE)
            user_index = None
            for idx, user_obj in enumerate(org_config["policy"]["allowedUserIds"]):
                if user_obj.get("userId") == user_id:
                    user_index = idx
                    break
            
            # Use atomic patch operation to increment the credit
            patch_operations = [
                {
                    "op": "incr",
                    "path": f"/policy/allowedUserIds/{user_index}/currentUsed",
                    "value": credit_consumed
                }
            ]

            updated_config = container.patch_item(
                item=org_config["id"],
                partition_key=organization_id,
                patch_operations=patch_operations
            )

            logging.info(
                f"[CosmosDB] Incremented user {user_id} credit by {credit_consumed}"
            )
            return updated_config

        except Exception as e:
            logging.error(f"[CosmosDB] Error updating user credit: {e}", exc_info=True)
            return None