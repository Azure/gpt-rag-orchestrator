from shared.util import get_chat_history_as_messages, replace_doc_ids_with_filepath, get_secret
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import logging
import openai
import openai.error
import os
import requests
import time
import tiktoken

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)

# Azure Cognitive Search Integration Settings

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "3"
AZURE_SEARCH_APPROACH = os.environ.get("AZURE_SEARCH_APPROACH") or "hybrid"
AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH") or "false"
AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = True if AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH == "true" else False
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG") or "my-semantic-config"
AZURE_SEARCH_SEMANTIC_SEARCH_LANGUAGE = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_LANGUAGE") or "en-us"
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN") or "true"
AZURE_SEARCH_ENABLE_IN_DOMAIN =  True if AZURE_SEARCH_ENABLE_IN_DOMAIN == "true" else False
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS") or "content"
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN") or "filepath"
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN") or "title"
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN") or "url"

# AOAI Integration Settings

AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-06-01-preview"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL") # 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") 
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1536"
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
ORCHESTRATOR_MESSAGES_LANGUAGE = os.environ.get("ORCHESTRATOR_MESSAGES_LANGUAGE") or "en"
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com"
openai.api_version = AZURE_OPENAI_API_VERSION
azureOpenAIKey = get_secret('azureOpenAIKey')
openai.api_key = azureOpenAIKey

model_max_tokens = {
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}

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


# prompt files
QUESTION_ANSWERING_OYD_PROMPT_FILE = f"orc/prompts/question_answering.oyd.prompt"
QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.prompt"

def get_answer_oyd(prompt, history):
            
        # prompt
        prompt = open(QUESTION_ANSWERING_OYD_PROMPT_FILE, "r").read() 

        # obs: temporarily removing previous questions from the history because AOAI OYD API is repeating answers from previous questions.
        messages=get_chat_history_as_messages(history, include_previous_questions=False, include_last_turn=True)
        sources = ""
        search_query = ""
        
        # creating body, headers and endpoint for the aoai rest api request
        azureSearchKey = get_secret('azureSearchKey') 
        body = {
            "messages": messages,
            "temperature": float(AZURE_OPENAI_TEMPERATURE),
            "max_tokens": int(AZURE_OPENAI_RESP_MAX_TOKENS),
            "top_p": float(AZURE_OPENAI_TOP_P),
            "stop": None,
            "stream": SHOULD_STREAM,
            "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": azureSearchKey,
                    "indexName": AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentField": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN,
                            "urlField": AZURE_SEARCH_URL_COLUMN,
                            "filepathField": AZURE_SEARCH_FILENAME_COLUMN
                        },
                        "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN else False,
                        "topNDocuments": AZURE_SEARCH_TOP_K,
                        "queryType": "semantic" if AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH else "simple",
                        "semanticConfiguration": "default",
                        "roleInformation": prompt
                    
                    }
            }]
        }

        chatgpt_url = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_CHATGPT_DEPLOYMENT}"
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview" # obs: this is the only api version that works with the chat endpoint

        headers = {
            'Content-Type': 'application/json',
            'api-key': azureOpenAIKey,
            'chatgpt_url': chatgpt_url,
            'chatgpt_key': azureOpenAIKey
        }
        endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_CHATGPT_DEPLOYMENT}/extensions/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

        # calling gpt model to get the answer
        start_time = time.time()
        try:
            response = requests.post(endpoint, headers=headers, json=body)
            completion = response.json()
            answer = completion['choices'][0]['messages'][1]['content']
            search_tool_result = json.loads(completion['choices'][0]['messages'][0]['content'])
            citations = search_tool_result["citations"]
            answer = replace_doc_ids_with_filepath(answer, citations)
            sources =  ""   
            for citation in citations:
                sources = sources + citation['filepath'] + ": "+ citation['content'].strip() + "\n"
            search_query = search_tool_result["intent"]
        except Exception as e:
            error_message = str(e)
            answer = f'{get_message("ERROR_ANSWER")}. {error_message}'
            logging.error(f"[orchestrator] {answer}")
        response_time = time.time() - start_time
        logging.info(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")
        
        return prompt, answer, sources, search_query

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    embeddings = response['data'][0]['embedding']
    return embeddings

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
def get_answer_from_gpt(messages):
    answer = ""
    completion = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=int(AZURE_OPENAI_RESP_MAX_TOKENS)
    )
    answer = completion['choices'][0]['message']['content']
    return answer

def number_of_tokens(messages):
    prompt = json.dumps(messages)
    model = AZURE_OPENAI_CHATGPT_MODEL
    encoding = tiktoken.encoding_for_model(model.replace('gpt-35-turbo','gpt-3.5-turbo'))
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

def create_rag_prompt(chat_history_messages, search_results):    
    # prompt
    prompt = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read()             
    if len(search_results) > 0:
        sources = "\n".join(search_results)
    else:        
        sources = "\n There are no sources for this question, say you don't have the answer"
    prompt = prompt.format(sources=sources)
    messages = [
        {"role": "system", "content": prompt}   
    ]
    messages = messages + chat_history_messages
    max_tokens = model_max_tokens[AZURE_OPENAI_CHATGPT_MODEL] - int(AZURE_OPENAI_RESP_MAX_TOKENS)
    num_tokens = number_of_tokens(messages)
    
    # reduces the number of search_results until it fits in the max tokens
    if (num_tokens > max_tokens) and len(search_results) > 0:
        search_results.pop()
        messages, prompt, sources = create_rag_prompt(chat_history_messages, search_results)

    # check if it is necessary to remove history messages from the prompt to fit in the max tokens
    num_tokens = number_of_tokens(messages)
    while(num_tokens > max_tokens):
        if len(messages) <=2: # keep at least 2 messages (system and user question)
            break
        messages.pop(1)
        num_tokens = number_of_tokens(messages)

    return messages, prompt, sources

def call_gpt_model(messages):
        # calling gpt model to get the answer
    start_time = time.time()
    try:
        answer = get_answer_from_gpt(messages)
        response_time = time.time() - start_time
        logging.info(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")
    except Exception as e:
        error_message = str(e)
        answer = f'{get_message("ERROR_CALLING_GPT")} {error_message}'
        logging.error(f"[orchestrator] error when calling gpt to get the answer. {error_message}")
    return answer

def get_answer_hybrid_search(history, semantic_reranking=False):

    # 1) retrieving grounding documents
    search_results = []
    search_query = history[-1]['content']
    prompt = ""
    answer = ""
    sources = ""
    error_on_search = False
    try:
        start_time = time.time()
        embeddings_query = generate_embeddings(search_query)
        response_time = time.time() - start_time
        logging.info(f"[orchestrator] generated question embeddings. {response_time} seconds")
        azureSearchKey = get_secret('azureSearchKey') 
        start_time = time.time()
        body = {
            "vector": {
                "value": embeddings_query,
                "fields": "contentVector",
                "k": int(AZURE_SEARCH_TOP_K)
            },
            "search": search_query,
            "select": "title, content, url, filepath, chunk_id",
            "top": AZURE_SEARCH_TOP_K
        }
        headers = {
            'Content-Type': 'application/json',
            'api-key': azureSearchKey
        }
        search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"

        if semantic_reranking:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG
            body["queryLanguage"] = AZURE_SEARCH_SEMANTIC_SEARCH_LANGUAGE

        response = requests.post(search_endpoint, headers=headers, json=body)
        status_code = response.status_code
        if status_code >= 400:
            error_on_search = True
            error_message = f'Status code: {status_code}.'
            if response.text != "": error_message += f" Error: {response.text}."
            answer = f'{get_message("ERROR_SEARCHING_DOCUMENTS")} {error_message}'
            logging.error(f"[orchestrator] error {status_code} when searching documents. {error_message}")
        else:
            if response.json()['value']:
                    for doc in response.json()['value']:
                        search_results.append(doc['filepath'] + ": "+ doc['content'].strip() + "\n")    
                
        response_time = time.time() - start_time
        logging.info(f"[orchestrator] searched documents. {response_time} seconds")
    except Exception as e:
        error_on_search = True
        error_message = str(e)
        answer = f'{get_message("ERROR_SEARCHING_DOCUMENTS")} {error_message}'
        logging.error(f"[orchestrator] error when searching documents {error_message}")

    # 2) generating answers
    if not error_on_search:
        chat_history_messages = get_chat_history_as_messages(history, include_last_turn=True)
        messages, prompt, sources = create_rag_prompt(chat_history_messages, search_results)    
        answer = call_gpt_model(messages)

    sources = sources.strip() if len(sources) > 0 else ""
    return prompt, answer, sources, search_query