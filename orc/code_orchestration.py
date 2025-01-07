# imports
import json
import logging
import os
import semantic_kernel as sk
import time
from orc.plugins.Conversation.Triage.wrapper import triage
from orc.plugins.ResponsibleAI.wrapper import fairness
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import KernelPlugin
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message, get_last_messages,get_possitive_int_or_default
from shared.util import get_blocked_list, create_kernel, get_usage_tokens, escape_xml_characters,get_secret
import asyncio
import xml.sax.saxutils as saxutils

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'debug').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables

BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK") or "true"
BLOCKED_LIST_CHECK = True if BLOCKED_LIST_CHECK.lower() == "true" else False
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK") or "true"
GROUNDEDNESS_CHECK = True if GROUNDEDNESS_CHECK.lower() == "true" else False
RESPONSIBLE_AI_CHECK = os.environ.get("RESPONSIBLE_AI_CHECK") or "true"
RESPONSIBLE_AI_CHECK = True if RESPONSIBLE_AI_CHECK.lower() == "true" else False
SECURITY_HUB_CHECK = os.environ.get("SECURITY_HUB_CHECK") or "false"
SECURITY_HUB_CHECK = True if SECURITY_HUB_CHECK.lower() == "true" else False
SECURITY_HUB_AUDIT = os.environ.get("SECURITY_HUB_AUDIT") or "false"
SECURITY_HUB_AUDIT = True if SECURITY_HUB_AUDIT.lower() == "true" else False
CONVERSATION_METADATA = os.environ.get("CONVERSATION_METADATA") or "true"
CONVERSATION_METADATA = True if CONVERSATION_METADATA.lower() == "true" else False

AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
CONVERSATION_MAX_HISTORY = os.environ.get("CONVERSATION_MAX_HISTORY") or "3"
CONVERSATION_MAX_HISTORY = int(CONVERSATION_MAX_HISTORY)
ORCHESTRATOR_FOLDER = "orc"
PLUGINS_FOLDER = f"{ORCHESTRATOR_FOLDER}/plugins"
BOT_DESCRIPTION_FILE = f"{ORCHESTRATOR_FOLDER}/bot_description.prompt"
BING_RETRIEVAL = os.environ.get("BING_RETRIEVAL") or "false"
BING_RETRIEVAL = True if BING_RETRIEVAL.lower() == "true" else False
SEARCH_RETRIEVAL = os.environ.get("SEARCH_RETRIEVAL") or "true"
SEARCH_RETRIEVAL = True if SEARCH_RETRIEVAL.lower() == "true" else False
RETRIEVAL_PRIORITY = os.environ.get("RETRIEVAL_PRIORITY") or "search"
SEVERITY_THRESHOLD = os.environ.get("SEVERITY_THRESHOLD") or 3
APIM_ENABLED = os.environ.get("APIM_ENABLED") or "false"
APIM_ENABLED = True if APIM_ENABLED.lower() == "true" else False
if SECURITY_HUB_CHECK:
    SECURITY_HUB_THRESHOLDS=[get_possitive_int_or_default(os.environ.get("SECURITY_HUB_HATE_THRESHHOLD"), 0),get_possitive_int_or_default(os.environ.get("SECURITY_HUB_SELFHARM_THRESHHOLD"), 0),get_possitive_int_or_default(os.environ.get("SECURITY_HUB_SEXUAL_THRESHHOLD"), 0),get_possitive_int_or_default(os.environ.get("SECURITY_HUB_VIOLENCE_THRESHHOLD"), 0)]

async def get_answer(history, security_ids,conversation_id):

    #############################
    # INITIALIZATION
    #############################

    #initialize variables    

    answer_dict = {}
    prompt = "The prompt is only recorded for question-answering intents"
    answer = ""
    intents = []
    bot_description = open(BOT_DESCRIPTION_FILE, "r").read()
    search_query = ""
    sources = ""
    detected_language = ""
    bypass_nxt_steps = False  # flag to bypass unnecessary steps
    blocked_list = []
    security_check=""
    # conversation metadata
    conversation_plugin_answer = ""
    conversation_history_summary = ''
    triage_language = ''
    answer_generated_by = "none"
    prompt_tokens = 0
    completion_tokens = 0
    apim_key=None
    if APIM_ENABLED:
        apim_key = await get_secret("apimSubscriptionKey")
    # get user question
    messages = get_chat_history_as_messages(history, include_last_turn=True)
    ask = messages[-1]['content']

    logging.info(f"[code_orchest] starting RAG flow. {ask[:50]}")
    init_time = time.time()
    # create kernel
    kernel = await create_kernel(apim_key=apim_key)
    # create the arguments that will used by semantic functions
    arguments = KernelArguments()
    arguments["bot_description"] = bot_description
    arguments["ask"] = ask
    arguments["history"] = json.dumps(get_last_messages(messages, CONVERSATION_MAX_HISTORY), ensure_ascii=False)
    arguments["previous_answer"] = messages[-2]['content'] if len(messages) > 1 else ""
    # import RAG plugins
    conversationPluginTask = asyncio.create_task(asyncio.to_thread(kernel.add_plugin, KernelPlugin.from_directory(parent_directory=PLUGINS_FOLDER,plugin_name="Conversation")))
    retrievalPluginTask = asyncio.create_task(asyncio.to_thread(kernel.add_plugin, KernelPlugin.from_directory(parent_directory=PLUGINS_FOLDER,plugin_name="Retrieval")))
    raiNativePluginTask = asyncio.create_task(asyncio.to_thread(kernel.add_plugin, KernelPlugin.from_directory(parent_directory=f"{PLUGINS_FOLDER}/ResponsibleAI/Native",plugin_name="Filters")))
    if(SECURITY_HUB_CHECK):
        securityPluginTask = asyncio.create_task(asyncio.to_thread(kernel.add_plugin, KernelPlugin.from_directory(parent_directory=PLUGINS_FOLDER,plugin_name="Security")))
    if(RESPONSIBLE_AI_CHECK):
        raiPluginTask = asyncio.create_task(asyncio.to_thread(kernel.add_plugin,KernelPlugin.from_directory(parent_directory=f"{PLUGINS_FOLDER}/ResponsibleAI",plugin_name="Semantic")))

    #############################
    # GUARDRAILS (QUESTION)
    #############################
    
    # AOAI Content filter validator
    raiNativePlugin = await raiNativePluginTask
    filterResult = await kernel.invoke(raiNativePlugin["ContentFliterValidator"], KernelArguments(input=ask,apim_key=apim_key))
    if not (filterResult.value.passed):
        logging.info(f"[code_orchest] filtered content found in question: {ask}.")
        answer = get_message('BLOCKED_ANSWER')
        answer_generated_by = 'content_filters_check'
        bypass_nxt_steps = True

    if BLOCKED_LIST_CHECK:
        logging.debug(f"[code_orchest] blocked list check.")
        try:
            blocked_list = await get_blocked_list()
            ask_words=ask.lower().split()
            for blocked_word in blocked_list:
                if blocked_word in ask_words:
                    logging.info(f"[code_orchest] blocked word found in question: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    answer_generated_by = 'blocked_list_check'
                    bypass_nxt_steps = True
                    break
        except Exception as e:
            logging.error(f"[code_orchest] could not get blocked list. {e}")
    response_time =  round(time.time() - init_time,2)
    logging.info(f"[code_orchest] finished content filter and blocklist check. {response_time} seconds.")
    conversationPlugin= await conversationPluginTask

    if SECURITY_HUB_CHECK and not bypass_nxt_steps:
            try:
                logging.info(f"[code_orchest] checking question with security hub. question: {ask[:50]}")
                start_time = time.time()
                arguments["answer"] = answer
                security_hub_key=await get_secret("securityHubKey")
                securityPlugin = await securityPluginTask
                security_check = await kernel.invoke(securityPlugin["QuestionSecurityCheck"], KernelArguments(question=ask,security_hub_key=security_hub_key))
                check_results = security_check.value["results"]
                check_details = security_check.value["details"]
                # New checks based on the updated requirements
                all_passed = True
                for name, status in check_results.items():
                    if status.lower() != "passed":
                        all_passed = False
                        break
                all_below_threshold = all(category["severity"] <= SECURITY_HUB_THRESHOLDS[index] for index,category in check_details.get("categoriesAnalysis", []))
                any_blocklists_match = len(check_details.get("blocklistsMatch", [])) > 0
                if not all_passed or not all_below_threshold or any_blocklists_match:
                    logging.error(f"[code_orchest] failed security hub question checks. Details: {check_details}.")
                    answer=get_message('BLOCKED_ANSWER')
                    answer_dict['security_hub'] = 1
                    answer_generated_by = "security_hub"
                    bypass_nxt_steps = True
                else:
                    answer_dict['security_hub'] = 5
                
                response_time = round(time.time() - start_time, 2)
                logging.info(f"[code_orchest] finished security hub checks. {response_time} seconds.")
            except Exception as e:
                logging.error(f"[code_orchest] could not execute security hub checks. {e}")  
                function_result = await call_semantic_function(kernel, conversationPlugin["NotInSourcesAnswer"], arguments)
                answer = str(function_result)
                answer_dict['security_hub'] = 1
                answer_generated_by = "security_hub"
                bypass_nxt_steps = True     
    #############################
    # RAG-FLOW
    #############################
    if not bypass_nxt_steps:

        try:
            # detect language
            logging.debug(f"[code_orchest] detecting language")
            start_time = time.time()
            function_result = await call_semantic_function(kernel, conversationPlugin["DetectLanguage"], arguments)
            prompt_tokens += get_usage_tokens(function_result, 'prompt')
            completion_tokens += get_usage_tokens(function_result, 'completion')            
            detected_language = str(function_result)
            arguments["language"] = detected_language
            response_time = round(time.time() - start_time,2)
            logging.info(f"[code_orchest] finished detecting language: {detected_language}. {response_time} seconds.")

            # conversation summary
            logging.debug(f"[code_orchest] summarizing conversation")
            start_time = time.time()
            if arguments["history"] != '[]':
                function_result = await call_semantic_function(kernel, conversationPlugin["ConversationSummary"], arguments)
                prompt_tokens += get_usage_tokens(function_result, 'prompt')
                completion_tokens += get_usage_tokens(function_result, 'completion')            
                conversation_history_summary =  str(function_result)
            else:
                conversation_history_summary = ""
                logging.info(f"[code_orchest] first time talking no need to summarize.")
            arguments["conversation_summary"] = conversation_history_summary
            response_time = round(time.time() - start_time,2)
            logging.info(f"[code_orchest] finished summarizing conversation: {conversation_history_summary}. {response_time} seconds.")

            # triage (find intent and generate answer and search query when applicable)
            logging.debug(f"[code_orchest] checking intent. ask: {ask}")
            start_time = time.time()
            triage_dict = await triage(kernel, conversationPlugin, arguments)
            intents = triage_dict['intents']
            prompt_tokens += triage_dict["prompt_tokens"]
            completion_tokens += triage_dict["completion_tokens"]
            response_time = round(time.time() - start_time,2)
            logging.info(f"[code_orchest] finished checking intents: {intents}. {response_time} seconds.")

            # Handle question answering intent
            if set(intents).intersection({"follow_up", "question_answering"}):         
    
                search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask
                search_sources= ""
                bing_sources=""
                #run search retrieval function
                retrievalPlugin= await retrievalPluginTask
                if(SEARCH_RETRIEVAL):
                    search_function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], KernelArguments(input=search_query,apim_key=apim_key,security_ids=security_ids))
                    formatted_sources = search_function_result.value[:100].replace('\n', ' ')
                    escaped_sources = escape_xml_characters(search_function_result.value)
                    search_sources=escaped_sources
                    
                #run bing retrieval function
                if(BING_RETRIEVAL):
                    if APIM_ENABLED:
                        bing_api_key=apim_key
                    else:
                        bing_api_key=await get_secret("bingapikey")
                    bing_custom_config_id=await get_secret("bingCustomConfigID")
                    bing_function_result= await kernel.invoke(retrievalPlugin["BingRetrieval"], KernelArguments(input=search_query, bing_api_key=bing_api_key, bing_custom_config_id=bing_custom_config_id))
                    formatted_sources = bing_function_result.value[:100].replace('\n', ' ')
                    escaped_sources = escape_xml_characters(bing_function_result.value)
                    bing_sources=escaped_sources
                
                
                if(RETRIEVAL_PRIORITY=="search"):
                    sources=search_sources+bing_sources
                else:
                    sources=bing_sources+search_sources
                arguments["sources"] = sources
                # Generate the answer augmented by the retrieval
                logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
                start_time = time.time()                                                          
                arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
                function_result = await call_semantic_function(kernel, conversationPlugin["Answer"], arguments)
                answer =  str(function_result)
                conversation_plugin_answer = answer
                answer_generated_by = "conversation_plugin_answer"
                prompt_tokens += get_usage_tokens(function_result, 'prompt')
                completion_tokens += get_usage_tokens(function_result, 'completion')
                prompt = str(function_result.metadata['messages'][0])
                response_time =  round(time.time() - start_time,2)              
                logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")

            # Handle general intents
            elif set(intents).intersection({"about_bot", "off_topic"}):
                answer = triage_dict['answer']
                answer_generated_by = "conversation_plugin_triage"
                logging.info(f"[code_orchest] triage answer: {answer}")
                
            elif "greeting" in intents:
                answer = triage_dict['answer']
                answer_generated_by = "conversation_plugin_triage"
                logging.info(f"[code_orchest] triage answer: {answer}")

            elif intents == ["none"]:
                logging.info(f"[code_orchest] No intent found, review Triage function.")
                answer = get_message('NO_INTENT_ANSWER')
                answer_generated_by = "no_intent_found_check"                
                bypass_nxt_steps = True

        except Exception as e:
            logging.error(f"[code_orchest] exception when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: exception: {e}"
            answer_generated_by = "exception_rag_flow"
            bypass_nxt_steps = True

    #############################
    # GUARDRAILS (ANSWER)
    #############################
    if BLOCKED_LIST_CHECK and not bypass_nxt_steps:
        try:
            answer_words = answer.lower().split()
            for blocked_word in blocked_list:
                if blocked_word in answer_words:
                    logging.info(f"[code_orchest] blocked word found in answer: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    answer_generated_by = "blocked_word_check"
                    break
        except Exception as e:
            logging.error(f"[code_orchest] could not get blocked list. {e}")
    if GROUNDEDNESS_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
            try:
                logging.info(f"[code_orchest] checking if it is grounded. answer: {answer[:50]}")
                groundness_time = time.time()            
                arguments["answer"] = saxutils.escape(answer)                      
                function_result = await call_semantic_function(kernel, conversationPlugin["IsGrounded"], arguments)
                grounded =  str(function_result)
                prompt_tokens += get_usage_tokens(function_result, 'prompt')
                completion_tokens += get_usage_tokens(function_result, 'completion')            
                logging.info(f"[code_orchest] is it grounded? {grounded}.")  
                if grounded.lower() == 'no':
                    logging.info(f"[code_orchest] ungrounded answer: {answer}")
                    function_result = await call_semantic_function(kernel, conversationPlugin["NotInSourcesAnswer"], arguments)
                    prompt_tokens += get_usage_tokens(function_result, 'prompt')
                    completion_tokens += get_usage_tokens(function_result, 'completion')            
                    answer =  str(function_result)
                    answer_dict['gpt_groundedness'] = 1
                    answer_generated_by = "gpt_groundedness_check"
                    bypass_nxt_steps = True
                else:
                    answer_dict['gpt_groundedness'] = 5
                response_time =  round(time.time() - groundness_time,2)
                logging.info(f"[code_orchest] finished checking if it is grounded. {response_time} seconds.")
            except Exception as e:
                logging.error(f"[code_orchest] could not check answer is grounded. {e}")           

    if RESPONSIBLE_AI_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
            try:
                logging.info(f"[code_orchest] checking responsible AI (fairness). answer: {answer[:50]}")
                start_time = time.time()            
                arguments["answer"] = saxutils.escape(answer)
                raiPlugin= await raiPluginTask
                fairness_dict = await fairness(kernel, raiPlugin, arguments)
                fair = fairness_dict['fair']
                fairness_answer = fairness_dict['answer']
                prompt_tokens += fairness_dict["prompt_tokens"]
                completion_tokens += fairness_dict["completion_tokens"]
                logging.info(f"[code_orchest] responsible ai check. Is it fair? {fair}.")
                if not fair:
                    answer = fairness_answer
                    answer_generated_by = "rai_plugin_fairness"
                answer_dict['pass_rai_fairness_check'] = fair
                response_time =  round(time.time() - start_time,2)
                logging.info(f"[code_orchest] finished checking responsible AI (fairness). {response_time} seconds.")
            except Exception as e:
                logging.error(f"[code_orchest] could not check responsible AI (fairness). {e}")
                
    if SECURITY_HUB_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
            try:
                logging.info(f"[code_orchest] checking answer with security hub. answer: {answer[:50]}")
                start_time = time.time()
                arguments["answer"] = saxutils.escape(answer)
                securityPlugin = await securityPluginTask
                security_check = await kernel.invoke(securityPlugin["AnswerSecurityCheck"], KernelArguments(question=ask, answer=answer, sources=sources,security_hub_key=security_hub_key))
                check_results = security_check.value["results"]
                check_details = security_check.value["details"]
                # New checks based on the updated requirements
                all_passed = True
                for name, status in check_results.items():
                    if status.lower() != "passed":
                        if name!="groundedness":
                            all_passed = False
                            break
                        elif check_details.get("groundedness", {}).get("ungroundedPercentage", 1) > float(os.environ.get("SECURITY_HUB_UNGROUNDED_PERCENTAGE_THRESHHOLD",0)):
                            all_passed = False
                            break
                all_below_threshold = all(category["severity"] <= SECURITY_HUB_THRESHOLDS[index] for index,category in check_details.get("categoriesAnalysis", []))
                any_blocklists_match = len(check_details.get("blocklistsMatch", [])) > 0
                if not all_passed or not all_below_threshold or any_blocklists_match:
                    logging.error(f"[code_orchest] failed security hub answer checks. Details: {check_details}.")
                    function_result = await call_semantic_function(kernel, conversationPlugin["NotInSourcesAnswer"], arguments)
                    answer = str(function_result)
                    answer_dict['security_hub'] = 1
                    answer_generated_by = "security_hub_answer_check"
                    bypass_nxt_steps = True
                else:
                    answer_dict['security_hub'] = 5
                
                response_time = round(time.time() - start_time, 2)
                logging.info(f"[code_orchest] finished answer security hub checks. {response_time} seconds.")
            except Exception as e:
                logging.error(f"[code_orchest] could not execute answer security hub checks. {e}")
    answer_dict["user_ask"] = ask if not answer_generated_by == 'content_filters_check' else '<FILTERED BY MODEL>'
    answer_dict["answer"] = answer
    answer_dict["search_query"] = search_query
    # additional metadata for debugging
    if CONVERSATION_METADATA:
        answer_dict["intents"] = intents
        answer_dict["detected_language"] = detected_language
        answer_dict["answer_generated_by"] = answer_generated_by
        answer_dict["conversation_history_summary"] = conversation_history_summary
        answer_dict["conversation_plugin_answer"] = conversation_plugin_answer
        answer_dict["model"] = AZURE_OPENAI_CHATGPT_MODEL
        answer_dict["prompt_tokens"] = prompt_tokens
        answer_dict["completion_tokens"] = completion_tokens

    if SECURITY_HUB_AUDIT:
        logging.info(f"[code_orchest] security hub audit.")
        await kernel.invoke(securityPlugin["Auditing"], KernelArguments(question=ask, answer=answer, sources=sources,security_hub_key=security_hub_key,conversation_id=conversation_id,security_checks=str(security_check)))
        
    answer_dict["prompt"] = prompt
    answer_dict["sources"] = sources.replace('[', '{').replace(']', '}')

    response_time =  round(time.time() - init_time,2)
    logging.info(f"[code_orchest] finished RAG Flow. {response_time} seconds.")

    return answer_dict