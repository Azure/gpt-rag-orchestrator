# utility functions

import re
import urllib.parse
import logging
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)

# KEY VAULT 

def get_secret(secretName):
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    logging.info(f"[orchestrator] retrieving {secretName} secret from {keyVaultName}.")   
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