import json
import logging
import os
import openai.error
import requests
import time
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from azure.search.documents.models import QueryType
from datetime import datetime
from shared.util import get_chat_history_as_text, get_chat_history_as_messages, get_aoai_call_data, get_completion_text, prompt_tokens, format_answer, replace_doc_ids_with_filepath

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)

# constants set from environment variables (external services credentials and configuration)

AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_GPT35TURBO_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT35TURBO_DEPLOYMENT") or "chat"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"

AZURE_DB_KEY = os.environ.get("AZURE_DB_KEY")
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER")

# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K")
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN")
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE")
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P")
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM")
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# prompt files
QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.prompt"

# predefined answers
THROTTLING_ANSWER = "Lo siento, nuestros servidores están demasiado ocupados, por favor inténtelo de nuevo en 10 segundos"
ERROR_ANSWER = "Lo siento, tuvimos un problema con la solicitud"

ANSWER_FORMAT = "html" # html, markdown, none


def run(conversation_id, ask):

    # 1) Create and configure service clients (cosmos, search and openai)

    # db
    db_client = CosmosClient(AZURE_DB_URI, credential=AZURE_DB_KEY, consistency_level='Session')

    # 2) Get conversation stored in CosmosDB

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id")

    # state mgmt
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
    # conversation_data
    conversation_data = conversation.get('conversation_data', 
                                    {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'interactions': [], 'aoai_calls': []})
    # transaction_data
    transactions = conversation.get('transactions', [])

    # 3) Define state/intent based on conversation and last statement from the user 

    # TODO: define state/intent based on conversation and last statement from the user to support transactional scenarios
    current_state = "question_answering" # let's stick to question answering for now
    conversation['state'] = current_state
    conversation = container.replace_item(item=conversation, body=conversation)

    # 4) Use conversation functions based on state

    # Initialize iteration variables
    answer = "none"
    sources = "none"
    search_query = "none"
    transaction_data_json = {}

    if current_state == "question_answering":
    # 4.1) Question answering

        prompt = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read() 

        messages=get_chat_history_as_messages(history, include_last_turn=True)
        
        # creating body, headers and endpoint for the aoai rest api request

        body = {
            "messages": messages,
            "temperature": float(AZURE_OPENAI_TEMPERATURE),
            "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
            "top_p": float(AZURE_OPENAI_TOP_P),
            "stop": None,
            "stream": SHOULD_STREAM,
            "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentField": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN,
                            "urlField": AZURE_SEARCH_URL_COLUMN,
                            "filepathField": AZURE_SEARCH_FILENAME_COLUMN
                        },
                        "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                        "topNDocuments": AZURE_SEARCH_TOP_K,
                        "queryType": "semantic" if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" else "simple",
                        "semanticConfiguration": "default",
                        "roleInformation": prompt
                    
                    }
            }]
        }

        chatgpt_url = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}"
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview"

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_KEY,
            'chatgpt_url': chatgpt_url,
            'chatgpt_key': AZURE_OPENAI_KEY
        }
        endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"
   
        # calling gpt model to get the answer

        start_time = time.time()

        try:
            response = requests.post(endpoint, headers=headers, json=body)
            status_code = response.status_code
            completion = response.json()
            answer = completion['choices'][0]['messages'][1]['content']
            search_tool_result = json.loads(completion['choices'][0]['messages'][0]['content'])
            citations = search_tool_result["citations"]
            answer = replace_doc_ids_with_filepath(answer, citations)
            sources =  ""   
            for citation in citations:
                sources = sources + citation['filepath'] + ": "+ citation['content'].strip() + "\n"
            search_query = search_tool_result["intent"]
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))
        except Exception as e:
            error_message = str(e)
            answer = f'{ERROR_ANSWER}. {error_message}'
            logging.error(f"[orchestrator] {answer}")
            conversation_data['aoai_calls'].append(get_aoai_call_data(messages, completion))

        response_time = time.time() - start_time
        logging.info(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")

    # 6. update and save conversation (containing history and conversation data)

    history.append({"role": "assistant", "content": answer})
    conversation['history'] = history
   
    conversation_data['interactions'].append({'user_message': ask, 'previous_state': previous_state, 'current_state': current_state})
    conversation['conversation_data'] = conversation_data

    conversation['transactions'] = transactions

    conversation = container.replace_item(item=conversation, body=conversation)

    # 7. return answer

    result = {"conversation_id": conversation_id, 
              "answer": format_answer(answer, ANSWER_FORMAT), 
              "current_state": current_state, 
              "data_points": sources, 
              "thoughts": f"Searched for:\n{search_query}\n\nPrompt:\n{prompt}",
              "transaction_data": transaction_data_json}
    return result