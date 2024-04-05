import os
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import uuid
import logging

AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

def store_user_consumed_tokens (user_id, consumed_tokens):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('userTokens')
        try:
            item = container.read_item(item=user_id, partition_key=user_id)
            existing_tokens = item['consumed_tokens']
        except Exception:
            # If the item doesn't exist, start with 0 tokens
            existing_tokens = 0
        total_tokens = existing_tokens + consumed_tokens
        container.upsert_item({
            'id': user_id,
            'consumed_tokens': total_tokens
        })
    except Exception as e:
        logging.error(f"Error retrieving the conversations: {e}")

def store_prompt_information (user_id, prompt_information):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('prompts')
        container.create_item({
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'prompt_information': prompt_information
        })
    except Exception as e:
        logging.error(f"Error retrieving the conversations: {e}")
