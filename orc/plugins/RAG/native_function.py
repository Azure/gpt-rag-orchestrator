from shared.util import get_secret, get_aoai_config
from semantic_kernel.skill_definition import sk_function
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import logging
import openai
import os
import re
import requests
import time

# Azure OpenAI Integration Settings
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")

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


# Set up logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):

    embeddings_config = get_aoai_config(AZURE_OPENAI_EMBEDDING_MODEL)

    openai.api_type = "azure_ad"
    openai.api_base = embeddings_config['endpoint']
    openai.api_version = embeddings_config['api_version']
    openai.api_key =  embeddings_config['api_key']

    response = openai.Embedding.create(
        input=text, engine=embeddings_config['deployment'])
    embeddings = response['data'][0]['embedding']
    return embeddings

class RAG:
    @sk_function(
        description=re.sub('\s+', ' ',f"""
            Search a knowledge base for sources to ground and give context to answer a user question. 
            Search in '{AZURE_SEARCH_SEMANTIC_SEARCH_LANGUAGE}' language. 
            Return sources."""),
        name="Retrieval",
        input_description="The user question",
    )
    def Retrieval(self, input: str) -> str:
        search_results = []
        search_query = input
        try:
            start_time = time.time()
            embeddings_query = generate_embeddings(search_query)
            response_time =  round(time.time() - start_time,2)
            logging.info(f"[sk_function_retrieval] querying azure ai search. search query: {search_query}")
            logging.info(f"[sk_function_retrieval] generated question embeddings. {response_time} seconds")
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
                logging.error(f"[sk_function_retrieval] error {status_code} when searching documents. {error_message}")
            else:
                if response.json()['value']:
                        for doc in response.json()['value']:
                            search_results.append(doc['filepath'] + ": "+ doc['content'].strip() + "\n")    
                    
            response_time =  round(time.time() - start_time,2)
            # logging.info(f"[sk_function_retrieval] search query body: {body}")        
            logging.info(f"[sk_function_retrieval] searched documents. {response_time} seconds")
        except Exception as e:
            error_message = str(e)
            logging.error(f"[sk_function_retrieval] error when getting the answer {error_message}")
        
        sources = ' '.join(search_results)
        return sources
        response_data = {"sources": sources}
        return json.dumps(response_data)