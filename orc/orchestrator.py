import re
import logging
import os
import time
import uuid
from azure.cosmos.aio import CosmosClient
from datetime import datetime
from shared.util import get_setting
from shared.cosmos_db import store_user_consumed_tokens, store_prompt_information
from azure.identity.aio import DefaultAzureCredential
import orc.code_orchestration as code_orchestration

from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

def get_credentials():
    is_local_env = os.getenv('LOCAL_ENV') == 'true'
    # return DefaultAzureCredential(exclude_managed_identity_credential=is_local_env, exclude_environment_credential=is_local_env)
    return DefaultAzureCredential()

def get_settings(client_principal):
    # use cosmos to get settings from the logged user
    data = get_setting(client_principal)
    temperature = 0.0 if 'temperature' not in data else data['temperature']
    frequency_penalty = 0.0 if 'frequencyPenalty' not in data else data['frequencyPenalty']
    presence_penalty = 0.0 if 'presencePenalty' not in data else data['presencePenalty']
    settings = {
        'temperature': temperature,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty
    }
    logging.info(f"[orchestrator] settings: {settings}")
    return settings

def instanciate_messages(messages_data):
    messages = []
    try:
        for message_data in messages_data:
            if message_data['type'] == 'human':
                message = HumanMessage(**message_data)
            elif message_data['type'] == 'system':
                message = SystemMessage(**message_data)
            elif message_data['type'] == 'ai':
                message = AIMessage(**message_data)
            else:
                Exception(f"Message type {message_data['type']} not recognized.")
                message.from_dict(message_data)
            messages.append(message)
        return messages
    except Exception as e:
        logging.error(f"[orchestrator] error instanciating messages: {e}")
        return []

def replace_numbers_with_paths(text, paths):
    citations = re.findall(r"\[([0-9]+(?:,[0-9]+)*)\]", text)
    for citation in citations:
        citation = citation.split(',')
        for c in citation:
            c = int(c)
            text = text.replace(f"[{c}]", "["+paths[c-1]+"]")
    logging.info(f"[orchestrator] response with citations {text}")
    return text

async def run(conversation_id, ask, client_principal):
    
    start_time = time.time()

    # 1) Get conversation stored in CosmosDB
 
    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id.")

    logging.info(f"[orchestrator] {conversation_id} starting conversation flow.")

    # get conversation
    credential = get_credentials()

    # settings
    settings = get_settings(client_principal)

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
                {'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'history': [
                    {'role': 'assistant', 'content': 'You are FreddAid, a friendly marketing assistant dedicated to uncovering insights and developing effective strategies.'}],
                    'messages_data': [{ 'type': 'system', 'content': 'You are FreddAid, a friendly marketing assistant dedicated to uncovering insights and developing effective strategies.',}], 'interaction': {}
                })
        # load messages data and instanciate them
        history = conversation_data['history']
        history.append({"role": "user", "content": ask})

        messages_data = conversation_data['messages_data']
        messages = instanciate_messages(messages_data)
        
        # 2) get answer and sources
        
        #TODO: apply settings
        #TODO: calculate consumend tokens
        #TODO: generate search query
        #TODO: store prompt information

        # get rag answer and sources
        logging.info(f"[orchestrator] executing RAG retrieval using code orchestration")
        with get_openai_callback() as cb:
            answer_dict = await code_orchestration.get_answer(ask, messages, settings)

        # 3) update and save conversation (containing history and conversation data)
        
        #messages data
        if 'human_message' in answer_dict:
            messages_data.append(answer_dict['human_message'].dict())
        if 'ai_message' in answer_dict:
            messages_data.append(answer_dict['ai_message'].dict())
        # history
        history.append({"role": "assistant", "content": answer_dict['answer']})
        conversation_data['history'] = history
        conversation_data['messages_data'] = messages_data

        # conversation data
        response_time = round(time.time() - start_time,2)
        interaction = {
            'user_id': client_principal['id'], 
            'user_name': client_principal['name'], 
            'response_time': response_time
        }
        conversation_data["interaction"] = interaction
        
        conversation['conversation_data'] = conversation_data
        conversation = await container.replace_item(item=conversation, body=conversation)

        # 4) store user consumed tokens

        store_user_consumed_tokens(client_principal['id'], cb)

        # 5) store prompt information in CosmosDB

        #store_prompt_information(client_principal['id'], answer_dict)
        answer_dict['answer'] = replace_numbers_with_paths(answer_dict['answer'], answer_dict['sources'])
        # 6) return answer
        result = {"conversation_id": conversation_id,
                "answer": answer_dict['answer'],
                "sources": answer_dict['sources'],
                "data_points": interaction['sources'] if 'sources' in interaction else '',
                "thoughts": ask #f"Searched for:\n{interaction['search_query']}\n\nPrompt:\n{interaction['prompt']}"
                }

        logging.info(f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds. answer: {answer_dict['answer'][:30]}")
        await db_client.close()

    return result