import logging
import openai
import os
import requests
import time
from shared.util import call_gpt_model, get_chat_history_as_messages, get_message, get_secret, number_of_tokens
from tenacity import retry, wait_random_exponential, stop_after_attempt

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Azure Cognitive Search Integration Settings

TERM_SEARCH_APPROACH='term'
VECTOR_SEARCH_APPROACH='vector'
HYBRID_SEARCH_APPROACH='hybrid'
AZURE_SEARCH_USE_SEMANTIC=os.environ.get("AZURE_SEARCH_USE_SEMANTIC")  or "false"
AZURE_SEARCH_APPROACH=os.environ.get("AZURE_SEARCH_APPROACH") or HYBRID_SEARCH_APPROACH

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "3"

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
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Additional Settings

QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.prompt"
model_max_tokens = {
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    embeddings = response['data'][0]['embedding']
    return embeddings

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

def get_answer(history):

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
        logging.info(f"[code_orchestration] generated question embeddings. {response_time} seconds")
        azureSearchKey = get_secret('azureSearchKey') 

        # prepare body
        body = {
            "select": "title, content, url, filepath, chunk_id",
            "top": AZURE_SEARCH_TOP_K
        }    
        if AZURE_SEARCH_APPROACH == TERM_SEARCH_APPROACH:
            body["search"] = search_query
        elif AZURE_SEARCH_APPROACH == VECTOR_SEARCH_APPROACH:
            body["vector"] = {
                "value": embeddings_query,
                "fields": "contentVector",
                "k": int(AZURE_SEARCH_TOP_K)
            }
        elif AZURE_SEARCH_APPROACH == HYBRID_SEARCH_APPROACH:
            body["search"] = search_query
            body["vector"] = {
                "value": embeddings_query,
                "fields": "contentVector",
                "k": int(AZURE_SEARCH_TOP_K)
            }
        if AZURE_SEARCH_USE_SEMANTIC == "true" and AZURE_SEARCH_APPROACH != VECTOR_SEARCH_APPROACH:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG
            body["queryLanguage"] = AZURE_SEARCH_SEMANTIC_SEARCH_LANGUAGE

        headers = {
            'Content-Type': 'application/json',
            'api-key': azureSearchKey
        }
        search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"
        
        start_time = time.time()
        response = requests.post(search_endpoint, headers=headers, json=body)
        status_code = response.status_code
        if status_code >= 400:
            error_on_search = True
            error_message = f'Status code: {status_code}.'
            if response.text != "": error_message += f" Error: {response.text}."
            answer = f'{get_message("ERROR_SEARCHING_DOCUMENTS")} {error_message}'
            logging.error(f"[code_orchestration] error {status_code} when searching documents. {error_message}")
        else:
            if response.json()['value']:
                    for doc in response.json()['value']:
                        search_results.append(doc['filepath'] + ": "+ doc['content'].strip() + "\n")    
                
        response_time = time.time() - start_time
        logging.info(f"[code_orchestration] search query body: {body}")        
        logging.info(f"[code_orchestration] searched documents. {response_time} seconds")
    except Exception as e:
        error_on_search = True
        error_message = str(e)
        answer = f'{get_message("ERROR_SEARCHING_DOCUMENTS")} {error_message}'
        logging.error(f"[code_orchestration] error when searching documents {error_message}")

    # 2) generating answers
    if not error_on_search:
        chat_history_messages = get_chat_history_as_messages(history, include_last_turn=True)
        messages, prompt, sources = create_rag_prompt(chat_history_messages, search_results)    
        answer, completion = call_gpt_model(messages)

    sources = sources.strip() if len(sources) > 0 else ""
    return prompt, answer, sources, search_query, completion