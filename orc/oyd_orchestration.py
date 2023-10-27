import json
import logging
import openai
import os
import requests
import time
from shared.util import get_chat_history_as_messages, get_message, get_secret, replace_doc_ids_with_filepath

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Azure Cognitive Search Integration Settings

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "3"

AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH") or "false"
AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = True if AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH == "true" else False
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN") or "true"
AZURE_SEARCH_ENABLE_IN_DOMAIN =  True if AZURE_SEARCH_ENABLE_IN_DOMAIN == "true" else False
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS") or "content"
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN") or "filepath"
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN") or "title"
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN") or "url"


# AOAI Integration Settings

AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-07-01-preview"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1536"
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Additional Settings

QUESTION_ANSWERING_OYD_PROMPT_FILE = f"orc/prompts/question_answering.oyd.prompt"

def get_answer(history):
        
        openai.api_type = "azure"
        openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com"
        openai.api_version = AZURE_OPENAI_API_VERSION
        azureOpenAIKey = get_secret('azureOpenAIKey')
        openai.api_key = azureOpenAIKey

            
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
        chatgpt_url += "/chat/completions?api-version=2023-07-01-preview" # obs: this is the only api version that works with the chat endpoint

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
            logging.error(f"[gpt_utils] {answer}")
        response_time =  round(time.time() - start_time,2)
        logging.info(f"[gpt_utils] called gpt model. {response_time} seconds")

        answer_dict = {
            "prompt" : prompt,
            "answer" : answer,
            "search_query" : search_query,
            "sources": sources,
            "prompt_tokens" : completion['usage']['prompt_tokens'],
            "completion_tokens" : completion['usage']['completion_tokens']
        }
        
        return answer_dict