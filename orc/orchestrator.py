import logging
import os
import time
import uuid
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from datetime import datetime
from shared.util import format_answer
from azure.identity import DefaultAzureCredential
import orc.code_orchestration as code_orchestration
import orc.oyd_orchestration as oyd_orchestration
import orc.promptflow_orchestration as promptflow_orchestration


# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# orchestration approach
USE_OYD='oyd'
USE_PROMPT_FLOW = 'promptflow'
USE_CODE='code'
ORCHESTRATION_APPROACH=os.environ.get("ORCHESTRATION_APPROACH") or USE_CODE

# Constants set from environment variables (external services credentials and configuration)

# Cosmos DB
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

# AOAI
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
AZURE_OPENAI_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

ANSWER_FORMAT = "html" # html, markdown, none

def get_credentials():
    is_local_env = os.getenv('LOCAL_ENV') == 'true'
    # return DefaultAzureCredential(exclude_managed_identity_credential=is_local_env, exclude_environment_credential=is_local_env)
    return DefaultAzureCredential()

async def run(conversation_id, ask, client_principal):
    
    start_time = time.time()

    # 1) Get conversation stored in CosmosDB
 
    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    logging.info(f"[orchestrator] starting conversation flow. conversation_id {conversation_id}. ask: {ask}")   

    # get conversation
    credential = get_credentials()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('conversations')
    try:
        conversation = container.read_item(item=conversation_id, partition_key=conversation_id)
    except Exception as e:
        logging.info(f"[orchestrator] {conversation_id} customer sent an inexistent conversation_id, create new conversation_id")        
        conversation = container.create_item(body={"id": conversation_id})
    logging.info(f"[orchestrator] conversation {conversation_id} retrieved.")

    # get conversation data
    conversation_data = conversation.get('conversation_data', 
                                        {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': []})
   
    # history
    history = conversation.get('history', [])
    history.append({"role": "user", "content": ask})

    # 2) Question answering

    # get rag answer and sources
    if ORCHESTRATION_APPROACH == USE_PROMPT_FLOW:
        logging.info(f"[orchestrator] executing RAG using PromptFlow orchestration") 
        answer_dict = promptflow_orchestration.get_answer(history)

    elif ORCHESTRATION_APPROACH == USE_OYD:
        logging.info(f"[orchestrator] executing RAG using Azure OpenAI on your data feature orchestration") 
        answer_dict = oyd_orchestration.get_answer(history)

    else: # USE_CODE
        logging.info(f"[orchestrator] executing RAG retrieval using code orchestration")

        answer_dict = await code_orchestration.get_answer(history)

    # 5. update and save conversation (containing history and conversation data)
    
    # history
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
    conversation = container.replace_item(item=conversation, body=conversation)
    
    # 6. return answer
    result = {"conversation_id": conversation_id, 
              "answer": format_answer(interaction['answer'], ANSWER_FORMAT), 
              "data_points": interaction['sources'] if 'sources' in interaction else '', 
              "thoughts": f"Searched for:\n{interaction['search_query']}\n\nPrompt:\n{interaction['prompt']}"}

    logging.info(f"[orchestrator] ended conversation flow. conversation_id {conversation_id}. answer: {interaction['answer'][:30]}")
    logging.info(f"[orchestrator] {conversation_id} conversation flow duration: {response_time} seconds.")  
    return result