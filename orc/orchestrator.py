import logging
import os
import time
import uuid
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from datetime import datetime
from shared.util import format_answer, get_secret
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
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER") or "conversations"

# AOAI
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL") # 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
AZURE_OPENAI_CHATGPT_TURBO_DEPLOYMENT= os.environ.get("AZURE_OPENAI_CHATGPT_TURBO_DEPLOYMENT") or "chat"

ANSWER_FORMAT = "html" # html, markdown, none

def run(conversation_id, ask, client_principal):
    
    start_time = time.time()

    # 1) Get conversation stored in CosmosDB
    azureDBkey = get_secret('azureDBkey')  
    db_client = CosmosClient(AZURE_DB_URI, credential=azureDBkey, consistency_level='Session')

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    logging.info(f"[orchestrator] starting conversation flow. conversation_id {conversation_id}. ask: {ask}")   

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

    # get conversation data
    conversation_data = conversation.get('conversation_data', 
                                        {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': []})
   
    # history
    history = conversation.get('history', [])
    history.append({"role": "user", "content": ask})

    # 2) Define state/intent based on conversation and last statement from the user 

    # TODO: define state/intent based on context and last statement from the user to support transactional scenarios
    current_state = "question_answering" # state mgmt (not used yet) fixed to question_answering
    conversation['state'] = current_state
    conversation = container.replace_item(item=conversation, body=conversation)

    # 3) Use conversation functions based on state
    
    if current_state == "question_answering":
    # 3.1) Question answering

        # get rag answer and sources
        if ORCHESTRATION_APPROACH == USE_PROMPT_FLOW:
            logging.info(f"[orchestrator] executing RAG using PromptFlow orchestration") 
            answer_dict = promptflow_orchestration.get_answer(history)

        elif ORCHESTRATION_APPROACH == USE_OYD:
            logging.info(f"[orchestrator] executing RAG using Azure OpenAI on your data feature orchestration") 
            answer_dict = oyd_orchestration.get_answer(history)

        else: # USE_CODE
            logging.info(f"[orchestrator] executing RAG retrieval using code orchestration")
            answer_dict = code_orchestration.get_answer(history)

    # 4. Add conversation data

    # 5. update and save conversation (containing history and conversation data)
    
    # history
    history.append({"role": "assistant", "content": answer_dict['answer']})
    conversation['history'] = history

    # conversation data
    response_time = round(time.time() - start_time,2)
    interaction = {
        'user_id': client_principal['id'], 
        'user_name': client_principal['name'], 
        'user_message': ask, 'previous_state': previous_state,         
        'current_state': current_state, 
        'response_time': response_time, 
        'model': AZURE_OPENAI_CHATGPT_MODEL
    }
    interaction.update(answer_dict)
    conversation_data['interactions'].append(interaction)
    conversation['conversation_data'] = conversation_data
    conversation = container.replace_item(item=conversation, body=conversation)
    
    # 6. return answer
    result = {"conversation_id": conversation_id, 
              "answer": format_answer(answer_dict['answer'], ANSWER_FORMAT), 
              "current_state": current_state, 
              "data_points": answer_dict['sources'], 
              "thoughts": f"Searched for:\n{['search_query']}\n\nPrompt:\n{answer_dict['prompt']}"}
          
    logging.info(f"[orchestrator] ended conversation flow. conversation_id {conversation_id}. answer: {answer_dict['answer'][:50]}")    

    return result