import logging
import os
import re
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from shared.util import get_aoai_config, truncate_to_max_tokens, get_secret
from shared.util import number_of_tokens, call_semantic_function, load_sk_plugin

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Gpt Model
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")

def get_groundedness(sources, answer):
     gpt_groundedness = -1
     try:

          # truncate sources to not hit model max token
          extra_tokens = 1500 + number_of_tokens(answer, AZURE_OPENAI_CHATGPT_MODEL)  # prompt + answer
          sources = truncate_to_max_tokens(sources, extra_tokens, AZURE_OPENAI_CHATGPT_MODEL) 


          # call semantic function to calculate groundedness
          oai_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
          kernel = sk.Kernel()
          kernel.add_chat_service("chat_completion", AzureChatCompletion(oai_config['deployment'], oai_config['endpoint'], oai_config['api_key']))
          context = kernel.create_new_context()
          context['sources'] = sources
          context['answer'] = re.sub(r'\[.*?\]', '', answer)
          rag_plugin = load_sk_plugin('RAG', oai_config)

          semantic_response = call_semantic_function(rag_plugin["Groundedness"], context)

          if not semantic_response.error_occurred:
               if semantic_response.result.isdigit():
                    gpt_groundedness = int(semantic_response.result)  
                    logging.info(f"[code_orchestration] groundedness: {gpt_groundedness}.")
               else:
                    logging.error(f"[monitoring] could not calculate groundedness.")
          else:
               logging.error(f"[monitoring] could not calculate groundedness. {semantic_response.last_error_description}")

     except Exception as e:
          logging.error(f"[monitoring] could not calculate groundedness. {e}")
          
     return gpt_groundedness

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
                                   logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. Gpt_groundedness {gpt_groundedness} is not a valid integer between 1 and 5.")
                         else:
                                   logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. LLM Monitoring is not enabled.")
               if changed: 
                    conversation = container.replace_item(item=conversation, body=conversation)

                    
     except Exception as e:
          logging.error(f"[monitoring] could not run monitoring. Error: {e}")