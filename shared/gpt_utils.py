from azure.core.credentials import AzureKeyCredential
from openai import ChatCompletion
from shared.util import get_chat_history_as_messages, replace_doc_ids_with_filepath
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
import json
import logging
import openai
import openai.error
import os
import requests
import time

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)

# Azure Cognitive Search Integration Settings

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION") or "2023-07-01-Preview"
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "3"
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH") or "false"
AZURE_SEARCH_USE_SEMANTIC_SEARCH =  True if AZURE_SEARCH_USE_SEMANTIC_SEARCH == "true" else False
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG") or "default"
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN") or "true"
AZURE_SEARCH_ENABLE_IN_DOMAIN =  True if AZURE_SEARCH_ENABLE_IN_DOMAIN == "true" else False
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS") or "content"
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN") or "filepath"
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN") or "title"
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN") or "url"

# AOAI Integration Settings

AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-06-01-preview"
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL") or "chat"
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL") or "text-embedding-ada-002"
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1536"
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False
openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com"
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_KEY

# predefined answers

THROTTLING_ANSWER = "Lo siento, nuestros servidores están demasiado ocupados, por favor inténtelo de nuevo en 10 segundos."
ERROR_ANSWER = "Lo siento, tuvimos un problema con la solicitud."
NOT_FOUND_ANSWER = "Lo siento, no tengo información cargada de este tema."

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
                        "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN else False,
                        "topNDocuments": AZURE_SEARCH_TOP_K,
                        "queryType": "semantic" if AZURE_SEARCH_USE_SEMANTIC_SEARCH else "simple",
                        "semanticConfiguration": "default",
                        "roleInformation": prompt
                    
                    }
            }]
        }

        chatgpt_url = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}"
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview" # obs: this is the only api version that works with the chat endpoint

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_KEY,
            'chatgpt_url': chatgpt_url,
            'chatgpt_key': AZURE_OPENAI_KEY
        }
        endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

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
            answer = f'{ERROR_ANSWER}. {error_message}'
            logging.error(f"[orchestrator] {answer}")
        response_time = time.time() - start_time
        logging.debug(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")
        
        return prompt, answer, sources, search_query

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine=AZURE_OPENAI_EMBEDDING_MODEL)
    embeddings = response['data'][0]['embedding']
    return embeddings

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
def get_answer_from_gpt(messages):
    answer = ""
    completion = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS)
    )
    answer = completion['choices'][0]['message']['content']
    return answer

def get_answer_hybrid_search(history):
        
        # prompt
        prompt = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read() 

        # searching documents
        seach_results = []
        search_query = history[-1]['content']
        answer = NOT_FOUND_ANSWER
        try:
            
            start_time = time.time()
            embeddings_query = generate_embeddings(search_query)
            response_time = time.time() - start_time
            logging.debug(f"[orchestrator] generated question embeddings. {response_time} seconds")

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
                'api-key': AZURE_SEARCH_KEY
            }
            search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"

            response = requests.post(search_endpoint, headers=headers, json=body)
            status_code = response.status_code
            if status_code >= 400:
                logging.error(f"[orchestrator] error {status_code} when searching documents. Status code: {response.text}")
            else:
                if response.json()['value']:
                     for doc in response.json()['value']:
                        seach_results.append(doc['filepath'] + ": "+ doc['content'].strip() + "\n")    
                    
            response_time = time.time() - start_time
            logging.debug(f"[orchestrator] searched documents. {response_time} seconds")
        except Exception as e:
            error_message = str(e)
            logging.error(f"[orchestrator] error when searching documents. {error_message}")

        # create question answering prompt
        history_messages=get_chat_history_as_messages(history, include_last_turn=True)
        # ground question answering prompt with search_results
        if len(seach_results) > 0:
            sources = "\n".join(seach_results)
        else:
            sources = "\n There are no sources for this question, say you don't have the answer"
        prompt = prompt.format(sources=sources)
        messages = [
            {"role": "system", "content": prompt}   
        ]
        messages = messages + history_messages

        # calling gpt model to get the answer
        start_time = time.time()
        try:
            answer = get_answer_from_gpt(messages)
            response_time = time.time() - start_time
            logging.debug(f"[orchestrator] called gpt model to get the answer. {response_time} seconds")
        except Exception as e:
            error_message = str(e)
            logging.error(f"[orchestrator] error when calling gpt to get the aswer. {error_message}")

        
        search_results = sources.strip() if len(sources) > 0 else ""

        return prompt, answer, search_results, search_query