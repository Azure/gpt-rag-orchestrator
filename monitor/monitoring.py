import logging
import os
import re
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from shared.util import get_secret, chat_complete, truncate_to_max_tokens, number_of_tokens
from tenacity import retry, wait_random_exponential, stop_after_attempt

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

EVALUATION_GROUNDEDNESS_PROMPT_FILE = f"orc/prompts/evaluation_groundedness.prompt"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT") or AZURE_OPENAI_CHATGPT_DEPLOYMENT
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL") # 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
AZURE_OPENAI_CHATGPT_MONITORING_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MONITORING_MODEL") or AZURE_OPENAI_CHATGPT_MODEL

@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(10), reraise=True)
def get_groundedness(sources, answer):
    prompt = open(EVALUATION_GROUNDEDNESS_PROMPT_FILE, "r").read() 
    # top tokens to the max tokens allowed by the model
    extra_tokens = number_of_tokens(prompt) + number_of_tokens(answer) + 100 # prompt + answer + messages overhead
    sources = truncate_to_max_tokens(sources, extra_tokens, AZURE_OPENAI_CHATGPT_MONITORING_MODEL)
    prompt = prompt.format(context=sources, answer=answer)
    messages = [
        {"role": "system", "content": prompt}   
    ]
    completion = chat_complete(messages, [], function_call='none', deployment=AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT, model=AZURE_OPENAI_CHATGPT_MONITORING_MODEL)
    groundedness = completion['choices'][0]['message']['content']
    
    return groundedness

def run():
     logging.info(f"[monitoring] running monitoring routine")
     AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
     AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
     AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER") or "conversations"
     AZURE_OPENAI_CHATGPT_LLM_MONITORING = os.environ.get("AZURE_OPENAI_CHATGPT_LLM_MONITORING") or "false"
     llmMonitoring = True if AZURE_OPENAI_CHATGPT_LLM_MONITORING.lower() == "true" else False     
     azureDBkey = get_secret('azureDBkey')  
     db_client = CosmosClient(AZURE_DB_URI, credential=azureDBkey, consistency_level='Session')

     # get conversations
     db = db_client.create_database_if_not_exists(id=AZURE_DB_ID)
     container = db.create_container_if_not_exists(id=AZURE_DB_CONTAINER, partition_key=PartitionKey(path='/id', kind='Hash'))
     try:
          conversations = container.query_items(query="SELECT * FROM c WHERE c.conversation_data.interactions != null", enable_cross_partition_query=True)
          for conversation in conversations:
               # process iteractions that have not yet been processed
               it = 0
               changed = False
               for interaction in conversation.get('conversation_data').get('interactions'):
                    it += 1
                    logging.info(f"[monitoring] processing conversation {conversation.get('id')} iteration {it}")
                    if 'sources' in interaction and 'answer' in interaction and 'gpt_groundedness' not in interaction:
                         if llmMonitoring:
                              # groundedness metric
                              gpt_groundedness = get_groundedness(interaction['sources'], interaction['answer'])
                              if re.match(r'^[1-5]$', str(gpt_groundedness)):
                                   interaction['gpt_groundedness'] = gpt_groundedness
                                   changed = True
                                   logging.info(f"[monitoring] conversation {conversation.get('id')} iteration {it} gpt_groundedness is {gpt_groundedness}.")
                              else:
                                   logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. Gpt_groundedness {gpt_groundedness} is not a valid integer between 1 and 5..")
                         else:
                                   logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. LLmMonitoring is not enabled.")
               if changed: 
                    conversation = container.replace_item(item=conversation, body=conversation)

                    
     except Exception as e:
          logging.error(f"[monitoring] could not run monitoring. Error: {e}")

