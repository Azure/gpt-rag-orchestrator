import logging
import os
import re
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
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

AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER") or "conversations"
AZURE_OPENAI_CHATGPT_LLM_MONITORING = os.environ.get("AZURE_OPENAI_CHATGPT_LLM_MONITORING") or "false"

def get_groundedness(sources, answer):
     gpt_groundedness = -1
     try:

          # truncate sources to not hit model max token
          extra_tokens = 1500 + number_of_tokens(answer, AZURE_OPENAI_CHATGPT_MODEL)  # prompt + answer
          sources = truncate_to_max_tokens(sources, extra_tokens, AZURE_OPENAI_CHATGPT_MODEL) 

          # call semantic function to calculate groundedness
          oai_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
          kernel = sk.Kernel()
          kernel.add_chat_service("chat_completion", AzureChatCompletion(oai_config['deployment'], oai_config['endpoint'], oai_config['api_key'], ad_auth=True))
          context = kernel.create_new_context()
          context['sources'] = sources
          context['answer'] = re.sub(r'\[.*?\]', '', answer)
          rag_plugin = load_sk_plugin('RAG', oai_config)

          semantic_response = call_semantic_function(rag_plugin["IsGrounded"], context)

          if not semantic_response.error_occurred:
               grounded = semantic_response.result
               if grounded.lower() == 'no':
                    gpt_groundedness= 1
               else:
                    gpt_groundedness = 5
          else:
               logging.error(f"[monitoring] could not calculate groundedness. Semantic Kernel: {semantic_response.last_error_description}")

     except Exception as e:
          logging.error(f"[monitoring] could not calculate groundedness. Error: {e}")
          
     return gpt_groundedness

def run():
     logging.info(f"[monitoring] running monitoring routine")
     llmMonitoring = True if AZURE_OPENAI_CHATGPT_LLM_MONITORING.lower() == "true" else False
     if llmMonitoring:
          credential = DefaultAzureCredential()
          db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
          # get conversations
          db = db_client.get_database_client(database=AZURE_DB_NAME)
          container = db.get_container_client('conversations')
          try:
               query = """
                    SELECT * FROM c WHERE c.conversation_data.interactions != null
                    AND EXISTS(
                    SELECT true  
                    FROM interaction IN c.conversation_data.interactions  
                    WHERE NOT IS_DEFINED (interaction.gpt_groundedness)
                    )
               """
               conversations = container.query_items(query=query, enable_cross_partition_query=True)
               for conversation in conversations:
                    # process iteractions that have not yet been processed
                    it = 0
                    changed = False
                    for interaction in conversation.get('conversation_data').get('interactions'):
                         it += 1
                         logging.info(f"[monitoring] processing conversation {conversation.get('id')} iteration {it}")
                         if 'sources' in interaction and 'answer' in interaction and 'gpt_groundedness' not in interaction:
                              if interaction['sources'] != '':
                                   gpt_groundedness = get_groundedness(interaction['sources'], interaction['answer'])
                                   if re.match(r'^[0-5]$', str(gpt_groundedness)):
                                        interaction['gpt_groundedness'] = gpt_groundedness
                                        changed = True
                                        logging.info(f"[monitoring] conversation {conversation.get('id')} iteration {it} gpt_groundedness is {gpt_groundedness}.")
                                   else:
                                        logging.warning(f"[monitoring] skipping conversation {conversation.get('id')} iteration {it}. Gpt_groundedness {gpt_groundedness} is not a valid integer between 1 and 5.")
                              else:
                                        interaction['gpt_groundedness'] = 0
                                        changed = True
                                        logging.warning(f"[monitoring] setting conversation {conversation.get('id')} iteration {it} groundedness to 0. It has no sources.")
                    if changed: 
                         conversation = container.replace_item(item=conversation, body=conversation)
          except Exception as e:
               logging.error(f"[monitoring] could not run monitoring. Error: {e}")