import logging
import os
import re
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from shared.util import get_secret, call_gpt_model
from tenacity import retry, wait_random_exponential, stop_after_attempt

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

EVALUATION_GROUNDNESS_PROMPT_FILE = f"orc/prompts/evaluation_groundness.prompt"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT") or AZURE_OPENAI_CHATGPT_DEPLOYMENT

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(6), reraise=True)
def get_groundness(sources, answer):
    prompt = open(EVALUATION_GROUNDNESS_PROMPT_FILE, "r").read() 
    prompt = prompt.format(context=sources, answer=answer)
    messages = [
        {"role": "system", "content": prompt}   
    ]
    groundness, completion = call_gpt_model(messages, deployment=AZURE_OPENAI_CHATGPT_MONITORING_DEPLOYMENT)
    groundness = completion['choices'][0]['message']['content']
    return groundness

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
          conversations = container.query_items(query="SELECT * FROM c WHERE ARRAY_LENGTH(c.conversation_data.interactions) > 0 AND ARRAY_CONTAINS(c.conversation_data.interactions, {'processed': false}, true)", enable_cross_partition_query=True)
          for conversation in conversations:
               # process iteractions that have not yet been processed
               it = 0
               changed = False
               for interaction in conversation.get('conversation_data').get('interactions'):
                    it += 1
                    logging.info(f"[monitoring] processing conversation {conversation.get('id')} iteration {it}")
                    if 'processed' not in interaction or not interaction['processed']:
                         if 'sources' in interaction and 'answer' in interaction:
                              if llmMonitoring:
                                   # groundness metric
                                   gpt_groundness = get_groundness(interaction['sources'], interaction['answer'])
                                   if re.match(r'^[1-5]$', str(gpt_groundness)):
                                        interaction['gpt_groundness'] = gpt_groundness
                                        interaction['processed'] = True
                                        changed = True
                                        logging.info(f"[monitoring] conversation {conversation.get('id')} iteration {it} gpt_groundness is {gpt_groundness}.")
                                   else:
                                        logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. Gpt_groundness {gpt_groundness} is not a valid integer between 1 and 5..")
                              else:
                                        logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. LLmMonitoring is not enabled.")
                    else:
                         logging.info(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. Iteration alredy processed.")
               if changed: 
                    conversation = container.replace_item(item=conversation, body=conversation)

                    
     except Exception as e:
          logging.error(f"[monitoring] could not get conversations from CosmosDB. Error: {e}")

