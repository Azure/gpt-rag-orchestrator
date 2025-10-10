import os
from zoneinfo import ZoneInfo
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import uuid
import logging
from datetime import datetime, timezone


AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

def store_agent_error(user_id, error, ask):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level="Session")
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client("agentErrors")
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
        logging.error(f"Error retrieving the conversations: {e}")

def get_conversation_data(conversation_id, user_id, type = None, user_timezone=None):
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential=credential)
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client("conversations")

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
            f"[CosmosDB] customer sent an inexistent conversation_id, saving new {conversation_id}"
        )
        # Initialize with default structure including all required keys
        conversation = container.create_item(
            body={
                "id": conversation_id,
                "user_id": user_id,
                "conversation_data": {
                    "start_date": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
                    "history": [],  # Initialize empty history list
                    "memory_data": "",
                    "interaction": {},
                    "type": type if type else "default",
                },
            }
        )

    # Get conversation data with complete default structure
    conversation_data = conversation.get(
        "conversation_data",
        {
            "start_date": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
            "history": [],  # Ensure history exists in default structure
            "memory_data": "",
            "interaction": {},
            "type": type if type else "default",
        },
    )

    # Double-check that history exists
    if "history" not in conversation_data:
        logging.info(
            f"[CosmosDB] 'history' not found for conversation_id: {conversation_id}. Initializing."
        )
        conversation_data["history"] = []
        
        # Update the conversation item in CosmosDB to include 'history'
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


def update_conversation_data(conversation_id, user_id, conversation_data):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential=credential)
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client("conversations")
        container.upsert_item(
            {
                "id": conversation_id,
                "user_id": user_id,
                "conversation_data": conversation_data,
            }
        )
    except Exception as e:
        logging.error(f"[CosmosDB] Error updating the conversations: {e}")

    return conversation_data

