import logging
import os
import time
import uuid
from azure.cosmos.aio import CosmosClient
from datetime import datetime
from shared.util import format_answer
from azure.identity.aio import ManagedIdentityCredential, AzureCliCredential, ChainedTokenCredential
import orc.code_orchestration as code_orchestration


# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

# Constants set from environment variables (external services credentials and configuration)

# Cosmos DB
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

# AOAI
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
AZURE_OPENAI_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

ANSWER_FORMAT = "html" # html, markdown, none

async def get_credentials():
    async with ChainedTokenCredential(
                ManagedIdentityCredential(),
                AzureCliCredential()
            ) as credential:
        return credential
    
def generate_security_ids(client_principal):
    security_ids = 'anonymous'
    if client_principal is not None:
        group_names = client_principal['group_names']
        security_ids = f"{client_principal['id']}" + (f",{group_names}" if group_names else "")
    return security_ids    
    
async def run(conversation_id, ask, client_principal):
    
    start_time = time.time()

    # 1) Get conversation stored in CosmosDB
 
    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id.")

    logging.info(f"[orchestrator] {conversation_id} starting conversation flow.")

    # get conversation

    #credential = get_credentials()
    async with ChainedTokenCredential(
                ManagedIdentityCredential(),
                AzureCliCredential()
            ) as credential:       
        async with CosmosClient(AZURE_DB_URI, credential=credential) as db_client:
            db = db_client.get_database_client(database=AZURE_DB_NAME)
            container = db.get_container_client('conversations')
            try:
                conversation = await container.read_item(item=conversation_id, partition_key=conversation_id)
                logging.info(f"[orchestrator] conversation {conversation_id} retrieved.")
            except Exception as e:
                logging.info(f"[orchestrator] customer sent an inexistent conversation_id, saving new conversation_id")        
                conversation = await container.create_item(body={"id": conversation_id})

            # get conversation data
            conversation_data = conversation.get('conversation_data', 
                                                {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': []})
        
            # history
            history = conversation.get('history', [])
            history.append({"role": "user", "content": ask})

            # 2) get answer and sources

            logging.info(f"[orchestrator] executing RAG retrieval using code orchestration")
            security_ids = generate_security_ids(client_principal)
            answer_dict = await code_orchestration.get_answer(history, security_ids,conversation_id)

            # 3) update and save conversation (containing history and conversation data)
            
            # history
            if answer_dict['answer_generated_by'] == 'content_filters_check': 
                history[-1]['content'] = '<FILTERED BY MODEL>'
            history.append({"role": "assistant", "content": answer_dict['answer']})
            conversation['history'] = history

            # conversation data
            response_time = round(time.time() - start_time,2)
            interaction = {
                'user_id': client_principal['id'], 
                'user_name': client_principal['name'], 
                'response_time': response_time
            }
            interaction.update(answer_dict)
            conversation_data['interactions'].append(interaction)
            conversation['conversation_data'] = conversation_data
            conversation = await container.replace_item(item=conversation, body=conversation)
            
            # 4) return answer
            result = {"conversation_id": conversation_id, 
                    "answer": format_answer(interaction['answer'], ANSWER_FORMAT), 
                    "data_points": interaction['sources'] if 'sources' in interaction else '', 
                    "thoughts": f"Searched for:\n{interaction['search_query']}\n\nPrompt:\n{interaction['prompt']}"}

            logging.info(f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds. answer: {interaction['answer'][:30]}")


    return result