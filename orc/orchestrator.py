import logging
import os
import uuid
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from datetime import datetime
from shared.gpt_utils import get_answer_oyd, get_answer_hybrid_search
from shared.util import format_answer

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)

# constants set from environment variables (external services credentials and configuration)

# Cosmos DB
AZURE_DB_KEY = os.environ.get("AZURE_DB_KEY")
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER") or "conversations"

# AOAI Integration Settings
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False
AZURE_SEARCH_USE_OYD = os.environ.get("AZURE_SEARCH_USE_OYD") or "true"
AZURE_SEARCH_USE_OYD = True if AZURE_SEARCH_USE_OYD.lower() == "true" else False
AZURE_SEARCH_USE_VECTOR_SEARCH = os.environ.get("AZURE_SEARCH_USE_VECTOR_SEARCH") or "false"
AZURE_SEARCH_USE_VECTOR_SEARCH = True if AZURE_SEARCH_USE_VECTOR_SEARCH.lower() == "true" else False

# prompt files
QUESTION_ANSWERING_OYD_PROMPT_FILE = f"orc/prompts/question_answering.oyd.prompt"
QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.prompt"

ANSWER_FORMAT = "html" # html, markdown, none

def run(conversation_id, ask):

    # 1) Get conversation stored in CosmosDB
    db_client = CosmosClient(AZURE_DB_URI, credential=AZURE_DB_KEY, consistency_level='Session')

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    # initializing state mgmt (not used yet)
    previous_state = "none"
    current_state = "none"

    # get conversation
    db = db_client.create_database_if_not_exists(id=AZURE_DB_ID)
    container = db.create_container_if_not_exists(id=AZURE_DB_CONTAINER, partition_key=PartitionKey(path='/id', kind='Hash'))
    try:
        conversation = container.read_item(item=conversation_id, partition_key=conversation_id)
        previous_state = conversation.get('state')
    except Exception as e:
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} customer sent an inexistent conversation_id, create new conversation_id")        
        conversation = container.create_item(body={"id": conversation_id, "state": previous_state})
    logging.info(f"[orchestrator] {conversation_id} previous state: {previous_state}")

    # history
    history = conversation.get('history', [])
    history.append({"role": "user", "content": ask})

    # 2) Define state/intent based on conversation and last statement from the user 

    # TODO: define state/intent based on context and last statement from the user to support transactional scenarios
    current_state = "question_answering" # state mgmt (not used yet) fixed to question_answering
    conversation['state'] = current_state
    conversation = container.replace_item(item=conversation, body=conversation)

    # 3) Use conversation functions based on state

    # Initialize iteration variables
    answer = "none"
    sources = "none"
    search_query = "none"
    
    if current_state == "question_answering":
    # 3.1) Question answering

        # 3.1.1) Azure OpenAI On Your Data Feature
        if (AZURE_SEARCH_USE_OYD):
            prompt = open(QUESTION_ANSWERING_OYD_PROMPT_FILE, "r").read() 
            answer, sources, search_query = get_answer_oyd(prompt, history)  

        # 3.1.2) hybrid search (vector + BM25)
        elif (AZURE_SEARCH_USE_VECTOR_SEARCH):
            prompt = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read() 
            answer, sources, search_query = get_answer_hybrid_search(prompt, history)

        # 3.1.3) BM25 search 
        else:
            pass

    # 4. update and save conversation (containing history and conversation data)

    history.append({"role": "assistant", "content": answer})
    conversation['history'] = history

    conversation = container.replace_item(item=conversation, body=conversation)

    # 5. return answer

    result = {"conversation_id": conversation_id, 
              "answer": format_answer(answer, ANSWER_FORMAT), 
              "current_state": current_state, 
              "data_points": sources, 
              "thoughts": f"Searched for:\n{search_query}\n\nPrompt:\n{prompt}"}
    return result