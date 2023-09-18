# utility functions

import re
import json
import logging
import openai
import os
import tiktoken
import time
import urllib.parse
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from tenacity import retry, wait_random_exponential, stop_after_attempt

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Env variables
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL") # 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1536"
ORCHESTRATOR_MESSAGES_LANGUAGE = os.environ.get("ORCHESTRATOR_MESSAGES_LANGUAGE") or "en"

# KEY VAULT 

def get_secret(secretName):
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    logging.info(f"[util] retrieving {secretName} secret from {keyVaultName}.")   
    retrieved_secret = client.get_secret(secretName)
    return retrieved_secret.value

# HISTORY FUNCTIONS

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
        history_list.insert(0, {"role": h["role"], "content": h["content"]})
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

# GPT FUNCTIONS

def get_completion_text(completion):
    if 'text' in completion['choices'][0]:
        return completion['choices'][0]['text'].strip()
    else:
        return completion['choices'][0]['message']['content'].strip()

# generates gpt usage data for statistics
def get_aoai_call_data(prompt, completion):
    prompt_words = 0
    if isinstance(prompt, list):
        messages = prompt
        prompt = ""
        for m in messages:
            prompt += m['role'].replace('\n',' ')
            prompt += m['content'].replace('\n',' ')
        prompt_words = len(prompt.split())
    else:
        prompt = prompt.replace('\n',' ')
        prompt_words = len(prompt.split())

    return {"model": completion["model"], "prompt_words": prompt_words}

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
def get_answer_from_gpt(messages, deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    answer = ""
    completion = openai.ChatCompletion.create(
        engine=deployment,
        messages=messages,
        temperature=0.0,
        max_tokens=int(AZURE_OPENAI_RESP_MAX_TOKENS)
    )
    answer = completion['choices'][0]['message']['content']
    return answer, completion

def call_gpt_model(messages, deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT):
        # calling gpt model to get the answer
    start_time = time.time()
    completion = None
    try:
        answer, completion = get_answer_from_gpt(messages, deployment)
        response_time = time.time() - start_time
        logging.info(f"[util] called gpt model. {response_time} seconds")
    except Exception as e:
        error_message = str(e)
        answer = f'{get_message("ERROR_CALLING_GPT")} {error_message}'
        logging.error(f"[util] error when calling gpt. {error_message}")
    return answer, completion

def number_of_tokens(messages):
    prompt = json.dumps(messages)
    model = AZURE_OPENAI_CHATGPT_MODEL
    encoding = tiktoken.encoding_for_model(model.replace('gpt-35-turbo','gpt-3.5-turbo'))
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


# FORMATTING FUNCTIONS

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

# MESSAGES FUNCTIONS

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