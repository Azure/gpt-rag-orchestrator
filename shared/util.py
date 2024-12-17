# utility functions

import re
import json
import logging
import os
import tiktoken
import time
import urllib.parse
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.keyvault.secrets.aio import SecretClient as AsyncSecretClient
from azure.identity.aio import ManagedIdentityCredential, AzureCliCredential, ChainedTokenCredential
from tenacity import retry, wait_random_exponential, stop_after_attempt
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from bs4 import BeautifulSoup
import aiohttp


# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
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
BING_RETRIEVAL = os.environ.get("BING_RETRIEVAL") or "true"
BING_RETRIEVAL = True if BING_RETRIEVAL.lower() == "true" else False
SEARCH_RETRIEVAL = os.environ.get("SEARCH_RETRIEVAL") or "true"
SEARCH_RETRIEVAL = True if SEARCH_RETRIEVAL.lower() == "true" else False
RETRIEVAL_PRIORITY = os.environ.get("RETRIEVAL_PRIORITY") or "search"
SECURITY_HUB_CHECK = os.environ.get("SECURITY_HUB_CHECK") or "false"
SECURITY_HUB_CHECK = True if SECURITY_HUB_CHECK.lower() == "true" else False
APIM_ENABLED = os.environ.get("APIM_ENABLED") or "false"
APIM_ENABLED = True if APIM_ENABLED.lower() == "true" else False

model_max_tokens = {
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4o': 8192 
}

##########################################################
# KEY VAULT 
##########################################################

async def get_secret(secretName):
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    async with ChainedTokenCredential( ManagedIdentityCredential(), AzureCliCredential()) as credential:
        async with AsyncSecretClient(vault_url=KVUri, credential=credential) as client:
            retrieved_secret = await client.get_secret(secretName)
            value = retrieved_secret.value

    # Consider logging the elapsed_time or including it in the return value if needed
    return value

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
async def call_semantic_function(kernel, function, arguments):
    function_result = await kernel.invoke(function, arguments)
    return function_result

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
async def chat_complete(messages, functions, params={}, function_call='auto',apim_key=None):
    """  Return assistant chat response based on user query. Assumes existing list of messages """

    oai_config = await get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)

    messages = optmize_messages(messages, AZURE_OPENAI_CHATGPT_MODEL)

    url = f"{oai_config['endpoint']}/openai/deployments/{oai_config['deployment']}/chat/completions?api-version={oai_config['api_version']}"
    if(APIM_ENABLED):
        headers = {
            "Content-Type": "application/json",
            "api-key": apim_key
        }
    else:   
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer "+ oai_config['api_key'] 
        }

    data = {
        "messages": messages,
        "max_tokens": params.get("max_tokens", int(AZURE_OPENAI_RESP_MAX_TOKENS)) 
    }

    if not function_call == 'none' and len(functions) > 0:
        data["functions"] = functions
        data["function_call"] = function_call

    if function_call == 'auto':
        data['temperature'] = 0
    else:
        data['temperature'] = params.get("temperature", float(AZURE_OPENAI_TEMPERATURE))
        data['top_p'] = params.get("top_p", float(AZURE_OPENAI_TOP_P))

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as request:
            response=await request.json()
    
    response_time =  round(time.time() - start_time,2)
    logging.info(f"[util__module] called chat completion api in {response_time:.6f} seconds")

    return response

##########################################################
# FORMATTING FUNCTIONS
##########################################################

# enforce answer format to the desired format (html, markdown, none)
def format_answer(answer, format= 'none'):
    
    formatted_answer = answer
    
    if format == 'html':
        
        # Convert bold syntax (**text**) to HTMLFhtml
        
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


def escape_xml_characters(input_string):
    """
    Escapes special characters in a string for XML.

    Args:
    input_string (str): The string to escape.

    Returns:
    str: The escaped string.
    """
    # Mapping of special characters to their escaped versions
    escape_mappings = {
        "&": "&amp;",
        "\"": "&quot;",
        "'": "&apos;",
        "<": "&lt;",
        ">": "&gt;"
    }

    # Replace each special character with its escaped version
    for key, value in escape_mappings.items():
        input_string = input_string.replace(key, value)

    return input_string

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

def get_last_messages(messages, n):
    """
    This function returns the last n*2 messages from the provided list, excluding the last message.

    Parameters:
    messages (list): A list of messages.
    n (int): The number of pairs of messages to return.

    Returns:
    list: A list containing the last n*2 messages, excluding the last message. If the input list is empty or contains only one message, an empty list is returned.

    Note:
    This function assumes that a conversation consists of pairs of messages (a message and a response). Therefore, it returns n*2 messages to get n pairs of messages.
    """    
    # Check if messages is not empty and has more than one element
    if messages and len(messages) > 1:
        # Get the last N*2 messages (N pairs), excluding the last message
        last_conversations = messages[-(n*2+1):-1]
        return last_conversations
    else:
        return []

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

async def create_kernel(service_id='aoai_chat_completion',apim_key=None):
    kernel = sk.Kernel()
    chatgpt_config =await get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
    if APIM_ENABLED:
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
                deployment_name=chatgpt_config['deployment'],
                endpoint=chatgpt_config['endpoint'],
                api_version=chatgpt_config['api_version'],
                api_key=apim_key
            )
        )
    else:
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
                deployment_name=chatgpt_config['deployment'],
                endpoint=chatgpt_config['endpoint'],
                api_version=chatgpt_config['api_version'],
                ad_token=chatgpt_config['api_key']
            )
        )
    return kernel

def get_usage_tokens(function_result, token_type='total'):
    metadata = function_result.metadata['metadata']
    usage_tokens = 0
    if token_type == 'completion':
        usage_tokens = sum(item['usage'].completion_tokens for item in metadata if 'usage' in item)
    elif token_type == 'prompt':
        usage_tokens = sum(item['usage'].prompt_tokens for item in metadata if 'usage' in item)
    elif token_type == 'total':
        usage_tokens = sum(item['usage'].total_tokens for item in metadata if 'usage' in item)        
    return usage_tokens

##########################################################
# AOAI FUNCTIONS
##########################################################

def get_list_from_string(string):
    result = string.split(',')
    result = [item.strip() for item in result]
    return result

async def get_aoai_config(model):
    if APIM_ENABLED:
        if model in ('gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k','gpt-4o'):
            deployment = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "gpt-4o"
        elif model == AZURE_OPENAI_EMBEDDING_MODEL:
            deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        else:
            raise Exception(f"Model {model} not supported. Check if you have the correct env variables set.")
        result = {
            "endpoint": os.environ.get("APIM_AZURE_OPENAI_ENDPOINT"),
            "deployment": deployment,
            "model": model,  # ex: 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k', 'gpt-4o'
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-03-01-preview",
        }
    else:
        resource = await get_next_resource(model)
        async with ChainedTokenCredential( ManagedIdentityCredential(), AzureCliCredential()) as credential:
            token = await credential.get_token("https://cognitiveservices.azure.com/.default")

            if model in ('gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k','gpt-4o'):
                deployment = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "gpt-4o"
            elif model == AZURE_OPENAI_EMBEDDING_MODEL:
                deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            else:
                raise Exception(f"Model {model} not supported. Check if you have the correct env variables set.")
            result = {
                "resource": resource,
                "endpoint": f"https://{resource}.openai.azure.com",
                "deployment": deployment,
                "model": model,  # ex: 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k', 'gpt-4o'
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-03-01-preview",
                "api_key": token.token
            }

    return result

async def get_next_resource(model):
    # define resource
    resources = os.environ.get("AZURE_OPENAI_RESOURCE")
    resources = get_list_from_string(resources)

    if not AZURE_OPENAI_LOAD_BALANCING or model == AZURE_OPENAI_EMBEDDING_MODEL:
        return resources[0]
    else:
        start_time = time.time()
        async with ChainedTokenCredential( ManagedIdentityCredential(), AzureCliCredential()) as credential:
            async with AsyncCosmosClient(AZURE_DB_URI, credential) as db_client:
                db = db_client.get_database_client(database=AZURE_DB_NAME)
                container = db.get_container_client('models')
                try:
                    keyvalue = await container.read_item(item=model, partition_key=model)
                    # check if there's an update in the resource list and update cache
                    if set(keyvalue["resources"]) != set(resources):
                        keyvalue["resources"] = resources
                except Exception:
                    logging.info(f"[util__module] get_next_resource: first time execution (keyvalue store with '{model}' id does not exist, creating a new one).")
                    keyvalue = {
                        "id": model,
                        "resources": resources
                    }
                    keyvalue = await container.create_item(body=keyvalue)
                resources = keyvalue["resources"]

                # get the first resource and move it to the end of the list
                resource = resources.pop(0)
                resources.append(resource)

                # update cache
                keyvalue["resources"] = resources
                await container.replace_item(item=model, body=keyvalue)

        response_time = round(time.time() - start_time, 2)
        logging.info(f"[util__module] get_next_resource: model '{model}' resource {resource}. {response_time} seconds")
        return resource
    
##########################################################
# OTHER FUNCTIONS
##########################################################

async def get_blocked_list():
    blocked_list = []
    async with ChainedTokenCredential( ManagedIdentityCredential(), AzureCliCredential()) as credential:
        async with AsyncCosmosClient(AZURE_DB_URI, credential) as db_client:
            db = db_client.get_database_client(database=AZURE_DB_NAME)
            container = db.get_container_client('guardrails')
            try:
                key_value = await container.read_item(item='blocked_list', partition_key='blocked_list')
                blocked_list = key_value["blocked_words"]
                blocked_list = [word.lower() for word in blocked_list]
            except Exception as e:
                logging.info(f"[util__module] get_blocked_list: no blocked words list (keyvalue store with 'blocked_list' id does not exist).")
    return blocked_list

async def extract_text_from_html(web,session):
    async with session.get(web.url) as html_response:
        try:
            html_response.raise_for_status()
            text=await html_response.text()
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup.find_all('header'):
                tag.decompose()
            for tag in soup.find_all('footer'):
                tag.decompose()
            for tag in soup.find_all('form'):
                tag.decompose()
            # Extract visible text from the HTML
            texts = soup.stripped_strings
            visible_text = ' '.join(texts)
            return visible_text
        except Exception as e:
            logging.error(f"Failed to extract text from url {web.url}, using snipet from bing: {e}")
            return web.snippet
    
def get_possitive_int_or_default(var, default_value):
    try:
        var = int(var)
        if var < 0:
            var = default_value
    except:
        var = default_value
    return var