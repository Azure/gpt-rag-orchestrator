# utility functions

import re
import json
import logging
import os
import requests
import tiktoken
import time
import urllib.parse
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from tenacity import retry, wait_random_exponential, stop_after_attempt
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Env variables
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"
AZURE_OPENAI_LOAD_BALANCING = os.environ.get("AZURE_OPENAI_LOAD_BALANCING") or "false"
AZURE_OPENAI_LOAD_BALANCING = True if AZURE_OPENAI_LOAD_BALANCING.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
ORCHESTRATOR_MESSAGES_LANGUAGE = os.environ.get("ORCHESTRATOR_MESSAGES_LANGUAGE") or "en"
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

model_max_tokens = {
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}

##########################################################
# KEY VAULT 
##########################################################

def get_secret(secretName):
    start_time = time.time()
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    retrieved_secret = client.get_secret(secretName)
    round(time.time() - start_time,2)
    logging.info(f"[util] get_secret: retrieving {secretName} secret from {keyVaultName}.")   
    return retrieved_secret.value

##########################################################
# HISTORY FUNCTIONS
##########################################################

def get_chat_history_as_text(history, include_last_turn=True, approx_max_tokens=1000):
    history_text = ""
    if len(history) == 0:
        return history_text
    for h in reversed(history if include_last_turn else history[:-1]):
        history_text = f"{h['role']}:" + h["content"]  + "\n" + history_text
        if len(history_text) > approx_max_tokens*4:
            break    
    return history_text

def get_chat_history_as_messages(history, include_previous_questions=True, include_last_turn=True, approx_max_tokens=1000):
    history_list = []
    if len(history) == 0:
        return history_list
    for h in reversed(history if include_last_turn else history[:-1]):
        history_item = {"role": h["role"], "content": h["content"]}
        if "function_call" in h:
            history_item.update({"function_call": h["function_call"]})
        if "name" in h:
            history_item.update({"name": h["name"]}) 
        history_list.insert(0, history_item)
        if len(history_list) > approx_max_tokens*4:
            break

    # remove previous questions if needed
    if not include_previous_questions:
        new_list = []
        for idx, item in enumerate(history_list):
            # keep only assistant messages and the last message
            # obs: if include_last_turn is True, the last user message is also kept 
            if item['role'] == 'assistant' or idx == len(history_list)-1:
                new_list.append(item)
        history_list = new_list        
    
    return history_list

##########################################################
# GPT FUNCTIONS
##########################################################

def number_of_tokens(messages, model):
    prompt = json.dumps(messages)
    encoding = tiktoken.encoding_for_model(model.replace('gpt-35-turbo','gpt-3.5-turbo'))
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

def truncate_to_max_tokens(text, extra_tokens, model):
    max_tokens = model_max_tokens[model] - extra_tokens
    tokens_allowed = max_tokens - number_of_tokens(text, model=model)
    while tokens_allowed < int(AZURE_OPENAI_RESP_MAX_TOKENS) and len(text) > 0:
        text = text[:-1]
        tokens_allowed = max_tokens - number_of_tokens(text, model=model)
    return text

# reduce messages to fit in the model's max tokens
def optmize_messages(chat_history_messages, model): 
    messages = chat_history_messages
    # check each get_sources function message and reduce its size to fit into the model's max tokens
    for idx, message in enumerate(messages):
        if message['role'] == 'function' and message['name'] == 'get_sources':
            # top tokens to the max tokens allowed by the model
            sources = json.loads(message['content'])['sources']

            tokens_allowed = model_max_tokens[model] - number_of_tokens(json.dumps(messages), model=model)
            while tokens_allowed < int(AZURE_OPENAI_RESP_MAX_TOKENS) and len(sources) > 0:
                sources = sources[:-1]
                content = json.dumps({"sources": sources})
                messages[idx]['content'] = content                
                tokens_allowed = model_max_tokens[model] - number_of_tokens(json.dumps(messages), model=model)

    return messages
   
@retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6), reraise=True)
def call_semantic_function(function, context):
    semantic_response = function(context = context)
    if semantic_response.error_occurred:
        error_code = 'none'
        if hasattr(semantic_response.last_exception, 'error_code'):
            error_code = str(semantic_response.last_exception.error_code)
        error_details = f"Error code: {error_code}. Error message: {semantic_response.last_error_description}"
        logging.info(f"[call_semantic_function] error occurred when calling semantic function {function.name}. {error_details}")
        if error_code == 'ErrorCodes.ServiceError':
            # TODO: add time penalty for model when service is unavailable
            pass
        raise Exception(f"Semantic function {function.name} failed with error: {semantic_response.last_error_description}")
    return semantic_response

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def chat_complete(messages, functions, function_call='auto'):
    """  Return assistant chat response based on user query. Assumes existing list of messages """

    oai_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)

    messages = optmize_messages(messages, AZURE_OPENAI_CHATGPT_MODEL)

    url = f"{oai_config['endpoint']}/openai/deployments/{oai_config['deployment']}/chat/completions?api-version={oai_config['api_version']}"

    headers = {
        "Content-Type": "application/json",
        # "api-key": oai_config['api_key']
        "Authorization": "Bearer "+ oai_config['api_key'] 
    }

    data = {
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "max_tokens": int(AZURE_OPENAI_RESP_MAX_TOKENS)
    }

    if function_call == 'auto':
        data['temperature'] = 0
    else:
        data['temperature'] = float(AZURE_OPENAI_TEMPERATURE)
        data['top_p'] = float(AZURE_OPENAI_TOP_P) 

    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    response_time =  round(time.time() - start_time,2)
    logging.info(f"[util] called chat completion api in {response_time:.6f} seconds")

    return response

##########################################################
# FORMATTING FUNCTIONS
##########################################################

# enforce answer format to the desired format (html, markdown, none)
def format_answer(answer, format= 'none'):
    
    formatted_answer = answer
    
    if format == 'html':
        
        # Convert bold syntax (**text**) to HTML
        formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_answer)
        
        # Convert italic syntax (*text*) to HTML
        formatted_answer = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_answer)
        
        # Return the converted text
    
    elif format == 'markdown':
        formatted_answer = answer # TODO
    
    elif format == 'none':        
        formatted_answer = answer # TODO

    return formatted_answer
  
# replace [doc1] [doc2] [doc3] with the corresponding filepath
def replace_doc_ids_with_filepath(answer, citations):
    for i, citation in enumerate(citations):
        filepath = urllib.parse.quote(citation['filepath'])
        answer = answer.replace(f"[doc{i+1}]", f"[{filepath}]")
    return answer

##########################################################
# MESSAGES FUNCTIONS
##########################################################

def get_message(message):
    if ORCHESTRATOR_MESSAGES_LANGUAGE.startswith("pt"):
        messages_file = "orc/messages/pt.json"
    elif ORCHESTRATOR_MESSAGES_LANGUAGE.startswith("es"):
        messages_file = "orc/messages/es.json"
    else:
        messages_file = "orc/messages/en.json"
    with open(messages_file, 'r') as f:
        json_data = f.read()
    messages_dict = json.loads(json_data)
    return messages_dict[message]

##########################################################
# SEMANTIC KERNEL
##########################################################

def load_sk_plugin(name, oai_config):
    kernel = sk.Kernel()
    kernel.add_chat_service("chat_completion", AzureChatCompletion(oai_config['deployment'], oai_config['endpoint'], oai_config['api_key'], ad_auth=True))
    plugin = kernel.import_semantic_skill_from_directory("orc/plugins", name)
    native_functions = kernel.import_native_skill_from_directory("orc/plugins", name)
    plugin.update(native_functions)
    return plugin


##########################################################
# AOAI FUNCTIONS
##########################################################

def get_list_from_string(string):
    result = string.split(',')
    result = [item.strip() for item in result]
    return result

def get_aoai_config(model):

    resource = get_next_resource(model)
    
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")

    if model in ('gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'):
        deployment = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
    elif model == AZURE_OPENAI_EMBEDDING_MODEL:
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    else:
        raise Exception(f"Model {model} not supported. Check if you have the correct env variables set.")

    result ={
        "resource": resource,
        "endpoint": f"https://{resource}.openai.azure.com",
        "deployment": deployment,
        "model": model, # ex: 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-08-01-preview",
        "api_key": token.token
    }
    return result

def get_next_resource(model):
    
    # define resource
    resources = os.environ.get("AZURE_OPENAI_RESOURCE")
    resources = get_list_from_string(resources)

    if not AZURE_OPENAI_LOAD_BALANCING or model == AZURE_OPENAI_EMBEDDING_MODEL:
        return resources[0]
    else:
        # get current resource list from cache
        start_time = time.time()
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('models')
        try:
            keyvalue = container.read_item(item=model, partition_key=model)
            # check if there's an update in the resource list and update cache
            if set(keyvalue["resources"]) != set(resources):
                keyvalue["resources"] = resources           
        except Exception:
            logging.info(f"[util] get_next_resource: first time execution (keyvalue store with '{model}' id does not exist, creating a new one).")  
            keyvalue = { 
                "id": model,
                "resources": resources              
            }      
            keyvalue = container.create_item(body=keyvalue)
        resources= keyvalue["resources"]

        # get the first resource and move it to the end of the list
        resource = resources.pop(0)
        resources.append(resource)

        # update cache
        keyvalue["resources"] = resources
        keyvalue = container.replace_item(item=model, body=keyvalue)

        response_time = round(time.time() - start_time,2)

        logging.info(f"[util] get_next_resource: model '{model}' resource {resource}. {response_time} seconds") 
        return resource
    
##########################################################
# OTHER FUNCTIONS
##########################################################

def get_blocked_list():
    start_time = time.time()
    blocked_list = []
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('guardrails')
    try:
        key_value = container.read_item(item='blocked_list', partition_key='blocked_list')
        blocked_list= key_value["blocked_words"]
        blocked_list = [word.lower() for word in blocked_list]  
    except Exception as e:
        logging.info(f"[util] get_blocked_list: no blocked words list (keyvalue store with 'blocked_list' id does not exist).")
    response_time = round(time.time() - start_time,2)
    logging.info(f"[util] get_blocked_list in {response_time} seconds") 
    return blocked_list