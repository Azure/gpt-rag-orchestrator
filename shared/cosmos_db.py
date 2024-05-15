import os
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import uuid
import logging
from datetime import datetime


AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

def store_user_consumed_tokens(user_id, consumed_tokens):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level="Session")
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client("userTokens")
        try:
            item = container.read_item(item=user_id, partition_key=user_id)
            existing_tokens = item.get("consumed_tokens", 0)
            existing_prompt_tokens = item.get("prompt_tokens", 0)
            existing_completion_tokens = item.get("completion_tokens", 0)
            existing_successful_requests = item.get("successful_requests", 0)
            existing_total_cost = item.get("total_cost", 0)
        except Exception:
            # If the item doesn't exist, start with 0
            existing_tokens = existing_prompt_tokens = existing_completion_tokens = (
                existing_successful_requests
            ) = existing_total_cost = 0
        container.upsert_item(
            {
                "id": user_id,
                "consumed_tokens": existing_tokens + consumed_tokens.prompt_tokens,
                "prompt_tokens": existing_prompt_tokens + consumed_tokens.prompt_tokens,
                "completion_tokens": existing_completion_tokens
                + consumed_tokens.completion_tokens,
                "successful_requests": existing_successful_requests
                + consumed_tokens.successful_requests,
                "total_cost": existing_total_cost + consumed_tokens.total_cost,
            }
        )
    except Exception as e:
        logging.error(f"Error retrieving the conversations: {e}")


def store_prompt_information(user_id, prompt_information):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level="Session")
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client("prompts")
        container.create_item(
            {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "prompt_information": prompt_information,
            }
        )
    except Exception as e:
        logging.error(f"Error retrieving the conversations: {e}")


def get_conversation_data(conversation_id):
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential=credential)
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client("conversations")

    try:
        conversation = container.read_item(
            item=conversation_id, partition_key=conversation_id
        )
    except Exception as e:
        logging.info(
            f"[CosmosDB] customer sent an inexistent conversation_id, saving new {conversation_id}"
        )
        conversation = container.create_item(body={"id": conversation_id})

    # get conversation data
    conversation_data = conversation.get(
        "conversation_data",
        {
            "start_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": [],
            "messages_data": [],
            "interaction": {},
        },
    )

    return conversation_data


def update_conversation_data(conversation_id, conversation_data):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential=credential)
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client("conversations")
        container.upsert_item(
            {
                "id": conversation_id,
                "conversation_data": conversation_data,
            }
        )
    except Exception as e:
        logging.error(f"[CosmosDB] Error updating the conversations: {e}")

    return conversation_data
