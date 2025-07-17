import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List
import aiohttp
import time
import asyncio
from functools import wraps
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from orc.graphs.tools import CustomRetriever, TavilySearch, retrieve_and_convert_to_document_format
from langgraph.checkpoint.memory import MemorySaver
from shared.prompts import (
    MARKETING_ORC_PROMPT,
    QUERY_REWRITING_PROMPT,
    AGUMENTED_QUERY_PROMPT, 
)
from langchain_core.runnables import RunnableParallel
from shared.cosmos_db import get_conversation_data
from shared.util import get_organization
from orc.graphs.tools.custom_agentic_search import generate_sub_queries

# Set up logging for Azure Functions - this needs to be done before creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # Override any existing logging configuration
)

# Configure the main module logger
logger = logging.getLogger(__name__)

# Configure Azure SDK specific loggers as per Azure SDK documentation
# Set logging level for Azure Search libraries
azure_search_logger = logging.getLogger("azure.search")
azure_search_logger.setLevel(logging.INFO)

# Set logging level for Azure Identity libraries
azure_identity_logger = logging.getLogger("azure.identity")
azure_identity_logger.setLevel(logging.WARNING)  # Less verbose for auth

# Set logging level for all Azure libraries (fallback)
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.WARNING)

# Suppress noisy Azure Functions worker logs
azure_functions_worker_logger = logging.getLogger("azure_functions_worker")
azure_functions_worker_logger.setLevel(logging.WARNING)

# Set logging level for LangChain libraries
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.WARNING)

# Set logging level for OpenAI libraries
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)

# Ensure propagation is enabled for Azure Functions
logger.propagate = True
azure_search_logger.propagate = True
azure_identity_logger.propagate = True
azure_logger.propagate = True
langchain_logger.propagate = True
openai_logger.propagate = True

# Document processing utilities
def _add_documents_to_dict(docs_dict: dict, documents: List[Document], source_type: str) -> int:
    """
    Add documents to deduplication dictionary with fallback key generation.
    
    Args:
        docs_dict: Dictionary for document deduplication
        documents: List of Document objects to add
        source_type: Type of source ('retrieval', 'web', 'adjacent') for fallback keys
        
    Returns:
        Number of documents added
    """
    added_count = 0
    
    for doc in documents:
        if hasattr(doc, 'id') and doc.id:
            docs_dict[doc.id] = doc
            added_count += 1
        else:
            # For documents without ID or with None ID, use a fallback key based on content hash
            fallback_key = f"{source_type}_{hash(doc.page_content)}"
            docs_dict[fallback_key] = doc
            added_count += 1
            logger.debug(f"[Document Processing] 🔧 Used fallback key for {source_type} doc: {fallback_key[:50]}...")
    
    return added_count

def _classify_document_source(doc: Document) -> str:
    """
    Classify document source type based on metadata.
    
    Args:
        doc: Document object to classify
        
    Returns:
        'web' if web search document, 'retrieval' otherwise
    """
    source = doc.metadata.get('source', '').lower() if hasattr(doc, 'metadata') else ''
    if 'http' in source and not any(domain in source for domain in ['blob.core.windows.net']):
        return 'web'
    return 'retrieval'

def _count_documents_by_source(docs: List[Document]) -> tuple:
    """
    Count documents by source type.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Tuple of (retrieval_count, web_count)
    """
    retrieval_count = 0
    web_count = 0
    
    for doc in docs:
        if _classify_document_source(doc) == 'web':
            web_count += 1
        else:
            retrieval_count += 1
    
    return retrieval_count, web_count

# initialize memory saver
@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_web_search: Flag indicating if web search is needed
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = field(default=False)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  # rewritten query for better search
    chat_summary: str = field(default_factory=str)
    token_count: int = field(default_factory=int)
    query_category: str = field(default_factory=str)
    agentic_search_mode: bool = field(default=True)
    augmented_query: str = field(default_factory=str)

def _truncate_chat_history(chat_history: List[dict], max_messages: int = 4) -> List[dict]:
    """
    Truncate chat history to the most recent messages.
    
    Args:
        chat_history: List of chat message dictionaries
        max_messages: Maximum number of messages to keep
        
    Returns:
        Truncated list of chat messages
    """
    if not chat_history:
        logger.info("[Chat History Cleaning] No chat history provided or empty list")
        return []
    
    logger.info(f"[Chat History Cleaning] Processing {len(chat_history)} messages")
    
    if len(chat_history) > max_messages:
        truncated_history = chat_history[-max_messages:]
        logger.info(f"[Chat History Cleaning] Truncated to last {max_messages} messages")
        return truncated_history
    else:
        logger.info(f"[Chat History Cleaning] Less than {max_messages} messages, no truncation needed")
        return chat_history

def clean_chat_history_for_agentic_search(chat_history: List[dict]) -> List[dict]:
    """
    Clean and format chat history for agentic search mode.
    
    Args:
        chat_history: List of chat message dictionaries
        
    Returns:
        List of formatted chat messages for agentic search
    """
    truncated_history = _truncate_chat_history(chat_history)
    if not truncated_history:
        return []
    
    formatted_history = [
        {"role": message.get("role", "").lower(), "content": message.get("content", "")} 
        for message in truncated_history
    ]
    logger.info(f"[Chat History Cleaning] Formatted for agentic search mode")
    return formatted_history

def clean_chat_history_for_llm(chat_history: List[dict]) -> str:
    """
    Clean and format chat history for LLM consumption as a string.
    
    Args:
        chat_history: List of chat message dictionaries
        
    Returns:
        Formatted chat history string in the format:
            Human: {message}
            AI Message: {message}
    """
    truncated_history = _truncate_chat_history(chat_history)
    if not truncated_history:
        return ""
    
    formatted_history = []
    for message in truncated_history:
        if not message.get("content"):
            continue

        role = message.get("role", "").lower()
        content = message.get("content", "")

        if role and content:
            display_role = "Human" if role == "user" else "AI Message"
            formatted_history.append(f"{display_role}: {content}")

    logger.info(f"[Chat History Cleaning] Formatted {len(formatted_history)} messages for LLM consumption")
    return "\n\n".join(formatted_history)

# Backward compatibility function
def clean_chat_history(chat_history: List[dict], agentic_search_mode: bool = False):
    """
    Clean the chat history and format it for consumption.
    
    Args:
        chat_history: List of chat message dictionaries
        agentic_search_mode: Whether to format for agentic search mode
        
    Returns:
        List[dict] for agentic search mode, str for LLM consumption
    """
    if agentic_search_mode:
        return clean_chat_history_for_agentic_search(chat_history)
    else:
        return clean_chat_history_for_llm(chat_history)

@dataclass
class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2025-01-01-preview"
    azure_deployment: str = "gpt-4.1"
    retriever_top_k: int = 5
    reranker_threshold: float = 2.1
    web_search_results: int = 2
    temperature: float = 0.4
    max_tokens: int = 50000 # input tokens could be really large
    agentic_search_mode: bool = False


class GraphBuilder:
    """Builds and manages the conversation flow graph."""
 
    def __init__( 
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
    ):
        """Initialize with with configuration"""
        logger.info(f"[GraphBuilder Init] Initializing GraphBuilder for conversation: {conversation_id}")
        logger.info(f"[GraphBuilder Init] Config - agentic_search_mode: {config.agentic_search_mode}, model temperature: {config.temperature}, max_tokens: {config.max_tokens}")
        
        self.organization_id = organization_id
        self.config = config
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.web_search = self._init_web_search()
        self.conversation_id = conversation_id
        self.organization_data = get_organization(organization_id)
        
        logger.info(f"[GraphBuilder Init] Successfully initialized GraphBuilder")

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        logger.info("[GraphBuilder LLM Init] Initializing Azure OpenAI client")
        config = self.config
        try:
            llm = AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=False,
                timeout=30,
                max_retries=3,
                azure_endpoint=os.getenv("O1_ENDPOINT"),
                api_key=os.getenv("O1_KEY")
            )
            logger.info(f"[GraphBuilder LLM Init] Successfully initialized Azure OpenAI with deployment: {config.azure_deployment}")
            return llm
        except Exception as e:
            logger.error(f"[GraphBuilder LLM Init] Failed to initialize Azure OpenAI: {str(e)}")
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")
    
    def _get_organization_data(self, data_key: str, data_name: str) -> str:
        """
        Retrieve organization data by key with consistent logging.
        
        Args:
            data_key: Key in organization_data dictionary
            data_name: Human-readable name for logging
            
        Returns:
            Organization data value or empty string if not found
        """
        data_value = self.organization_data.get(data_key, '')
        logger.info(f"[GraphBuilder {data_name} Init] Retrieved {data_name.lower()} (local memory) for organization {self.organization_id}")
        return data_value
    
    def _init_segment_alias(self) -> str:
        """Retrieve segment alias."""
        return self._get_organization_data('segmentSynonyms', '')
    
    def _init_brand_information(self) -> str:
        """Retrieve brand information."""
        return self._get_organization_data('brandInformation', '')
    
    def _init_industry_information(self) -> str:
        """Retrieve industry information."""
        return self._get_organization_data('industryInformation', '')

    def _build_organization_context_prompt(self, history: List[dict]) -> str:
        """
        Build the organization context prompt with conversation history and organization data.
        
        Args:
            history: List of conversation history messages
            
        Returns:
            Formatted organization context prompt
        """
        return f"""
        <-------------------------------->
        
        Historical Conversation Context:
        <-------------------------------->
        ```
        {clean_chat_history_for_llm(history)}
        ```
        <-------------------------------->

        **Alias segment mappings:**
        <-------------------------------->
        alias to segment mappings typically look like this (Official Name -> Alias):
        A -> B
        
        This mapping is mostly used in consumer segmentation context. 
        
        Critical Rule – Contextual Consistency with Alias Mapping:
    •	Always check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
    •	ALWAYS use the official name (A) in the rewritten query.
    •	DO NOT use the alias (B) in the rewritten query. 

        Here is the actual alias to segment mappings:
        
        **Official Segment Name Mappings (Official Name -> Alias):**
        ```
        {self._init_segment_alias()}
        ```

        For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

        Look, if a mapping in the instruction is like this:
        students -> young kids 

        Though the historical conversation and the orignal question may mention "students", you must rewrite the question to use "young kids" instead of "students".

        <-------------------------------->
        Brand Information:
        <-------------------------------->
        ```
        {self._init_brand_information()}
        ```
        <-------------------------------->

        Industry Information:
        <-------------------------------->
        ```
        {self._init_industry_information()}
        ```
        <-------------------------------->

        """

    def _init_retriever(self) -> CustomRetriever:
        logger.info("[GraphBuilder Retriever Init] Initializing Custom Agentic Retriever")
        try:
            config = self.config
            # index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
            index_name = "ragindex-test"
            if not index_name:
                logger.error("[GraphBuilder Retriever Init] AZURE_AI_SEARCH_INDEX_NAME is not set")
                raise ValueError(
                    "AZURE_AI_SEARCH_INDEX_NAME is not set in the environment variables" 
                )
            
            retriever = CustomRetriever(
                indexes=[index_name],
                topK=config.retriever_top_k,
                reranker_threshold=config.reranker_threshold,
                organization_id=self.organization_id,
            )
            logger.info(f"[GraphBuilder Retriever Init] Successfully initialized Custom Agentic Retriever with index: {index_name}, topK: {config.retriever_top_k}")
            return retriever
        except Exception as e:
            logger.error(f"[GraphBuilder Retriever Init] Failed to initialize Azure AI Search Retriever: {str(e)}")
            raise RuntimeError(
                f"Failed to initialize Azure AI Search Retriever: {str(e)}"
            )
    
    def _run_agentic_retriever(self, conversation_history: List[dict]):
        # Safe handling of potentially None conversation_history
        conversation_history = conversation_history or []
        logger.info(f"[GraphBuilder Agentic Retriever] Running agentic retriever with {len(conversation_history)} messages")
        results = retrieve_and_convert_to_document_format(conversation_history, self.organization_id)
        # Safe handling of potentially None results
        results_count = len(results) if results else 0
        logger.info(f"[GraphBuilder Agentic Retriever] Retrieved {results_count} documents")
        return results or []

    async def _execute_single_query_async(self, query_info: tuple, semaphore: asyncio.Semaphore = None) -> tuple:
        """Execute a single query retrieval asynchronously with rate limiting and optional web search.
        
        Args:
            query_info: Tuple of (query_index, query_text, query_type)
            semaphore: Optional semaphore for rate limiting
            
        Returns:
            Tuple of (query_index, query_type, query_text, retrieval_results, web_results, execution_time)
        """
        query_index, query_text, query_type = query_info
        start_time = time.time()
        
        # Use semaphore for rate limiting if provided
        async with (semaphore if semaphore else asyncio.Semaphore(4)):
            try:
                logger.info(f"[Async Custom Agentic Search] 🔍 Starting {query_type}: {query_text}")

                await asyncio.sleep(0.1)  # 100ms delay between requests
                
                # Execute retrieval
                loop = asyncio.get_event_loop()
                retrieval_results = await loop.run_in_executor(None, self.retriever.invoke, query_text)
                
                # Check if we need web search (less than 2 documents)
                web_results = []
                needs_web_search = len(retrieval_results) < 2
                
                if needs_web_search:
                    logger.info(f"[Async Custom Agentic Search] 🌐 {query_type} retrieved {len(retrieval_results)} documents - performing web search")
                    try:
                        # Perform web search for this specific query
                        web_results = await loop.run_in_executor(None, self.web_search.search, query_text)
                        logger.info(f"[Async Custom Agentic Search] 🌐 {query_type} web search completed - {len(web_results)} documents")
                    except Exception as web_error:
                        logger.error(f"[Async Custom Agentic Search] ❌ {query_type} web search failed: {str(web_error)}")
                        web_results = []
                else:
                    logger.info(f"[Async Custom Agentic Search] ✅ {query_type} retrieved {len(retrieval_results)} documents - no web search needed")
                
                execution_time = time.time() - start_time
                
                total_docs = len(retrieval_results) + len(web_results)
                logger.info(f"[Async Custom Agentic Search] ✅ {query_type} completed in {execution_time:.2f}s - {total_docs} total documents ({len(retrieval_results)} retrieval + {len(web_results)} web)")
                
                return (query_index, query_type, query_text, retrieval_results, web_results, execution_time)
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[Async Custom Agentic Search] ❌ {query_type} failed in {execution_time:.2f}s: {str(e)}")
                
                # Implement exponential backoff for capacity errors
                if "capacity" in str(e).lower() or "overload" in str(e).lower():
                    backoff_delay = min(2 ** (query_index - 1), 10)  # Max 10 seconds
                    logger.warning(f"[Async Custom Agentic Search] ⏳ Capacity issue detected, backing off for {backoff_delay}s")
                    await asyncio.sleep(backoff_delay)
                
                return (query_index, query_type, query_text, [], [], execution_time)

    async def _execute_queries_async(self, sub_queries: List[str]) -> tuple:
        """Execute all queries asynchronously with improved error handling and rate limiting.
        
        Args:
            sub_queries: List of query strings to execute
            
        Returns:
            Tuple of (docs_dict, total_retrieved, total_execution_time, parallel_total_time, any_web_search_used)
        """
        # Prepare queries for async execution
        query_tasks = []
        for i, query in enumerate(sub_queries, 1):
            query_type = "Rewritten Query" if i == len(sub_queries) else f"SUB-QUERY {i}"
            query_tasks.append((i, query, query_type))
        
        logger.info(f"[Async Custom Agentic Search] 🚀 EXECUTING {len(query_tasks)} QUERIES ASYNCHRONOUSLY")
        logger.info("\n" + "=" * 80)
        
        # Create semaphore for rate limiting (max 3 concurrent requests)
        semaphore = asyncio.Semaphore(3)
        
        # Execute queries in parallel
        docs_dict = {}
        total_retrieved = 0
        total_execution_time = 0
        
        parallel_start_time = time.time()
        
        # Create async tasks
        tasks = [
            self._execute_single_query_async(task, semaphore) 
            for task in query_tasks
        ]
        
        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[Async Custom Agentic Search] ❌ Task failed with exception: {result}")
                continue
                
            query_index, query_type, query_text, retriever_results, web_results, execution_time = result
            
            # Combine retrieval and web search results
            all_docs = retriever_results + web_results
            results_count = len(all_docs)
            total_retrieved += results_count
            total_execution_time += execution_time
            logger.info("\n" + "*" * 80)
            logger.info(f"\n[Async Custom Agentic Search] 📋 {query_type} RESULTS:")
            logger.info(f"[Async Custom Agentic Search] ├─ Query: {query_text}")
            logger.info(f"[Async Custom Agentic Search] ├─ Execution time: {execution_time:.2f}s")
            logger.info(f"[Async Custom Agentic Search] ├─ Total documents: {results_count}")
            logger.info(f"[Async Custom Agentic Search] ├─ Retrieval documents: {len(retriever_results)}")
            logger.info(f"[Async Custom Agentic Search] ├─ Web search documents: {len(web_results)}")
            
            # Log document sources if available
            if all_docs:
                logger.info(f"[Async Custom Agentic Search] ├─ 📄 All documents:")
                
                # Log retrieval documents first
                if retriever_results:
                    logger.info(f"[Async Custom Agentic Search] │  ├─ 🗄️ From retrieval:")
                    for j, doc in enumerate(retriever_results, 1):  
                        source = doc.metadata.get('source', 'Unknown source')
                        logger.info(f"[Async Custom Agentic Search] │  │  └─ {j}. Source: {source}")
                
                # Log web search documents
                if web_results:
                    logger.info(f"[Async Custom Agentic Search] │  └─ 🌐 From web search:")
                    for j, doc in enumerate(web_results, 1):  
                        source = doc.metadata.get('source', 'Unknown source')
                        logger.info(f"[Async Custom Agentic Search] │     └─ {j}. Source: {source}")
            else:
                logger.info(f"[Async Custom Agentic Search] ├─ ❌ No documents found for this query")
            
            logger.info(f"[Async Custom Agentic Search] └─ {query_type} PROCESSING COMPLETED")
            logger.info("\n" + "=" * 80)

            # Add documents to deduplication dictionary
            logger.info(f"[Async Custom Agentic Search] ├─ 🔄 Adding {len(all_docs)} documents to deduplication dict")
            
            retrieval_added = _add_documents_to_dict(docs_dict, retriever_results, "retrieval")
            web_added = _add_documents_to_dict(docs_dict, web_results, "web")
            
            logger.info(f"[Async Custom Agentic Search] ├─ 📊 Added to dict: {retrieval_added} retrieval + {web_added} web = {retrieval_added + web_added} total")
            logger.info(f"[Async Custom Agentic Search] ├─ 📋 Current dict size: {len(docs_dict)} unique documents")
        
        parallel_total_time = time.time() - parallel_start_time
        
        # Track if any query used web search
        any_web_search_used = False
        for result in results:
            if not isinstance(result, Exception):
                _, _, _, _, web_results, _ = result
                if len(web_results) > 0:
                    any_web_search_used = True
                    break
        
        logger.info(f"[Async Custom Agentic Search] 🔍 Web search usage detected: {'YES' if any_web_search_used else 'NO'}")
        
        return docs_dict, total_retrieved, total_execution_time, parallel_total_time, any_web_search_used

    def _execute_custom_agentic_search(self, original_query: str, rewritten_query: str, historical_conversation: str) -> tuple:
        """
        Execute custom agentic search with sub-queries and parallel processing.
        
        Args:
            rewritten_query: The query to generate sub-queries from
            
        Returns:
            Tuple of (docs_list, any_web_search_used)
        """
        # generate sub queries
        logger.info(f"[Custom Agentic Search Main] Generating sub-queries for retrieval")
        sub_queries = generate_sub_queries(original_query, rewritten_query, historical_conversation, self.llm)
        # create a list of sub queries, include the rewritten query 
        sub_queries = [query.strip() for query in sub_queries] + [rewritten_query]
        
        logger.info(f"[Custom Agentic Search Main] Generated {len(sub_queries)-1 if len(sub_queries) > 1 else len(sub_queries)} sub-queries for retrieval and added the rewritten query to the search list")
        logger.info("\n" + "-" * 78)
        logger.info("[Custom Agentic Search Main] Sub-queries generated:")
        for i, query in enumerate(sub_queries, 1):
            query_type = "Rewritten Query" if i == len(sub_queries) else f"SUB-QUERY {i}"
            logger.info(f"[Custom Agentic Search Main]   {query_type}: {query}")
        logger.info("\n" + "-" * 78)

        # Prepare queries for parallel execution
        query_tasks = []
        for i, query in enumerate(sub_queries, 1):
            query_type = "Rewritten Query" if i == len(sub_queries) else f"SUB-QUERY {i}"
            query_tasks.append((i, query, query_type))
        
        try:
            docs_dict, total_retrieved, total_execution_time, parallel_total_time, any_web_search_used = asyncio.run(
                self._execute_queries_async(sub_queries)
            )
                
        except Exception as e:
            logger.error(f"[Custom Agentic Search Main] ❌ Async execution failed: {e}")
            logger.error(f"[Custom Agentic Search Main] 💥 Async execution is required for custom agentic search")
            # Re-raise the exception rather than falling back
            raise RuntimeError(f"Async execution failed: {e}") from e

        # Convert dictionary values to list
        docs = list(docs_dict.values())
        unique_docs = len(docs)
        
        # Count final document sources for verification
        final_retrieval_count, final_web_count = _count_documents_by_source(docs)
        
        logger.info("\n" + "=" * 78)
        logger.info(f"[Custom Agentic Search Main] 📊 EXECUTION SUMMARY:")
        logger.info(f"[Custom Agentic Search Main] ├─ Total queries executed: {len(sub_queries)}")
        logger.info(f"[Custom Agentic Search Main] ├─ Execution time: {parallel_total_time:.2f}s")
        logger.info(f"[Custom Agentic Search Main] ├─ Total documents retrieved: {total_retrieved}")
        logger.info(f"[Custom Agentic Search Main] ├─ Unique documents after deduplication: {unique_docs}")
        logger.info(f"[Custom Agentic Search Main] ├─ 📋 Final composition: {final_retrieval_count} retrieval + {final_web_count} web documents")
        logger.info(f"[Custom Agentic Search Main] ├─ 🔍 Web search usage: {'Used by at least one query' if any_web_search_used else 'Not used by any query'}")
        logger.info(f"[Custom Agentic Search Main] ├─ 🎯 Per-query web search triggered for queries with <2 retrieval results")
        logger.info("\n" + "=" * 78)
        
        return docs, any_web_search_used

    def _init_web_search(self):
        logger.info("[GraphBuilder Web Search Init] Initializing Tavily web search")
        try:
            config = self.config
            # return GoogleSearch(k=config.web_search_results)
            web_search = TavilySearch(max_results=config.web_search_results)
            logger.info(f"[GraphBuilder Web Search Init] Successfully initialized TavilySearch with max_results: {config.web_search_results}")
            return web_search
        except Exception as e:
            logger.error(f"[GraphBuilder Web Search Init] Failed to initialize Tavily Web Search: {str(e)}")
            raise RuntimeError(f"Failed to initialize Tavily Web Search: {str(e)}")

    def _return_state(self, state: ConversationState) -> dict:
        # Safe handling of potentially None state
        messages_count = len(state.messages) if state.messages else 0
        context_docs_count = len(state.context_docs) if state.context_docs else 0
        
        logger.info(f"[GraphBuilder Return State] Returning state with {messages_count} messages, {context_docs_count} context docs")
        return {
            "messages": state.messages,
            "context_docs": state.context_docs,
            "chat_summary": state.chat_summary,
            "token_count": state.token_count,
            "requires_web_search": state.requires_web_search,
            "rewritten_query": state.rewritten_query,
            "query_category": state.query_category,
        }

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        logger.info("[GraphBuilder Build] Starting graph construction")
        # set up graph
        graph = StateGraph(ConversationState)
        # Add processing nodes

        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("tool_choice", self._categorize_query)
        graph.add_node("route", self._route_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("return", self._return_state)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "tool_choice")
        graph.add_edge("tool_choice", "route")
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {"retrieve": "retrieve", "return": "return"},
        )

        graph.add_edge("retrieve", "return")
        graph.add_edge("return", END)

        compiled_graph = graph.compile(checkpointer=memory)
        logger.info("[GraphBuilder Build] Successfully constructed conversation processing graph")
        return compiled_graph

    async def _rewrite_query(self, state: ConversationState) -> dict:
        logger.info(f"[Query Rewrite] Starting async query rewrite for: '{state.question[:100]}...'")
        question = state.question

        system_prompt = QUERY_REWRITING_PROMPT

        conversation_data = get_conversation_data(self.conversation_id)
        history = conversation_data.get("history", [])
        logger.info(f"[Query Rewrite] Retrieved {len(history)} messages from conversation history")

        # combine the system prompt with the additional system prompt
        system_prompt = f"{system_prompt}\n\n{self._build_organization_context_prompt(history)}"

        prompt = f"""Original Question: 
        <-------------------------------->
        ```
        {question}. 
        ```
        <-------------------------------->

        Please rewrite the question to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        ALSO, THE HISTORICAL CONVERSATION CONTEXT IS VERY VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, $10,000 WILL BE DEDUCTED FROM YOUR ACCOUNT IF YOU DO NOT USE THE HISTORICAL CONVERSATION CONTEXT.
        Please also consider the line of business/industry of my company when rewriting the query. Don't be too verbose. 

        if the question is a very casual/conversational one, do not rewrite, return it as it is
        """
        
        logger.info("[Query Rewrite] Sending async query rewrite request to LLM")
        rewritte_query = await self.llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )
        logger.info(f"[Query Rewrite] Successfully rewrote query: '{rewritte_query.content[:100]}...'")

        if state.messages is None:
            state.messages = []
        
        # augment the query with the historical conversation context
        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {clean_chat_history_for_llm(history)}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text formatonly, no additional text, explanations, or formatting.
        
        """
        logger.info(f"[Query Augment] Sending async augmented query request to LLM {augmented_query_prompt}")
        try:
            augmented_query = await self.llm.ainvoke(
                [SystemMessage(content=AGUMENTED_QUERY_PROMPT), HumanMessage(content=augmented_query_prompt)]
            )
            logger.info(f"[Query Augment] Successfully augmented query: '{augmented_query.content[:100]}...'")
        except Exception as e:
            logger.error(f"[Query Augment] Failed to augment query, using original question: {e}")
            augmented_query = question
        
        return {
            "rewritten_query": rewritte_query.content,
            "augmented_query": augmented_query.content,
            "messages": state.messages + [HumanMessage(content=question)],
        }

    async def _categorize_query(self, state: ConversationState) -> dict:
        """Categorize the query."""
        logger.info(f"[Query Categorization] Starting async query categorization for: '{state.question[:100]}...'")

        conversation_data = get_conversation_data(self.conversation_id)
        history = conversation_data.get("history", [])
        logger.info(f"[Query Categorization] Using {len(history)} conversation history messages for context")

        category_prompt = f"""
        You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - General

        Use both the current question and the historical conversation context to make an informed decision. 
        Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

        To help you make an accurate decision, consider these cues for each category:

        - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
        - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
        - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
        - **Creative Copywriter**: Use this category when the user asks for help creating or refining marketing text. This includes taglines, headlines, ad copy, email subject lines, social captions, website copy, or product descriptions. Trigger this if the user is brainstorming, writing, or editing text with a creative, promotional purpose.
        - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

        If the question or context is not clearly related to any of the above categories, always return "General".

        ----------------------------------------
        User's Question:
        {state.question}
        ----------------------------------------
        Conversation History:
        {clean_chat_history_for_llm(history)}
        ----------------------------------------

        Reply with **only** the exact category name — no additional text, explanations, or formatting.
        """
        
        logger.info("[Query Categorization] Sending async categorization request to LLM")
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=category_prompt),
                HumanMessage(content=state.question),
            ],
            temperature=0
        )
        logger.info(f"[Query Categorization] Categorized query as: '{response.content}'")

        return {
            "query_category": response.content
        }

    async def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""
        logger.info(f"[Query Routing] Determining async routing decision for query: '{state.rewritten_query[:100]}...'")

        system_prompt = MARKETING_ORC_PROMPT

        logger.info("[Query Routing] Sending async routing decision request to LLM")
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"How should I categorize this question: \n\n{state.rewritten_query}\n\nAnswer yes/no."
                ),
            ]
        )
        
        llm_suggests_retrieval = response.content.lower().startswith("y")
        logger.info(f"[Query Routing] LLM initial assessment - Not a casual/conversational question, proceed to retrieve documents: {llm_suggests_retrieval}")
        
        return {
            "requires_retrieval": llm_suggests_retrieval,
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        decision = "retrieve" if state.requires_retrieval else "return"
        logger.info(f"[Route Decision] Routing to: '{decision}' (requires_retrieval: {state.requires_retrieval})")
        return decision

    def _retrieve_context(self, state: ConversationState) -> dict:
        """Get relevant documents from vector store."""
        logger.info("\n" + "=" * 78)
        logger.info(f"[Retrieve Context] 🔍 STARTING CONTEXT RETRIEVAL PHASE")
        logger.info(f"[Retrieve Context] ├─ Search mode: {'Agentic Search' if self.config.agentic_search_mode else 'Custom Agentic Search'}")
        logger.info(f"[Retrieve Context] ├─ Rewritten query: {state.rewritten_query}")
        docs = []
        any_web_search_used = False 

        conversation_history = get_conversation_data(self.conversation_id).get("history", [])
                    # clean the conversation history
        conversation_history = clean_chat_history_for_agentic_search(conversation_history)

            # append the rewritten query to the conversation history
        conversation_history.append({"role": "user", "content": state.rewritten_query})
        if self.config.agentic_search_mode:

            docs = self._run_agentic_retriever(conversation_history)
            # For agentic search mode, we don't track per-query web search usage
            any_web_search_used = False

        else:
            docs, any_web_search_used = self._execute_custom_agentic_search(original_query=state.question, rewritten_query=state.rewritten_query, historical_conversation=conversation_history)
### <-------------------------------->
### Since we're adopting sub queries, let's turn off the adjacent chunks for now 
            # if docs:
            #     # print id of the first document in the list
            #     print(f'Document ID of the top ranked doc: {docs[0].id}')
            #     # get adjacent chunks
            #     adjacent_chunks = self.retriever._search_adjacent_pages(docs[0].id)
            #     # reformat adjacent chunks
            #     if adjacent_chunks:
            #         for chunk in adjacent_chunks:
            #             # Create Document object for adjacent chunk
            #             adjacent_doc = Document(
            #                 page_content=chunk['content'], 
            #                 metadata={'source': chunk['filepath'], 'score': "adjacent chunk"}
            #             )
                        
            #             # Add to dictionary (automatically handles deduplication)
            #             if hasattr(adjacent_doc, 'id') and adjacent_doc.id:
            #                 docs_dict[adjacent_doc.id] = adjacent_doc
            #             else:
            #                 # For adjacent chunks without ID, use a fallback key
            #                 fallback_key = f"adjacent_{hash(chunk['content'])}"
            #                 docs_dict[fallback_key] = adjacent_doc
                    
            #         print(f"total number of docs before adding adjacent chunks: {len(docs)}")
            #         # Update docs list with potentially new adjacent chunks
            #         docs = list(docs_dict.values())
            #         print(f"total number of docs after adding adjacent chunks: {len(docs)}")
### <-------------------------------->
        
        # Final decision based on actual retrieval results
        # If any individual query used web search, we mark it as web search needed
        web_search_needed = any_web_search_used if not self.config.agentic_search_mode else len(docs) < 2
        
        # Final verification of document sources in retrieved docs
        final_retrieval_docs, final_web_docs = _count_documents_by_source(docs)
        logger.info("\n" + "=" * 78)
        logger.info(f"[Retrieve Context] 📋 POST-RETRIEVAL DECISION ANALYSIS:")
        logger.info(f"[Retrieve Context] ├─ Documents retrieved: {len(docs)}")
        logger.info(f"[Retrieve Context] ├─ 📊 Final docs composition: {final_retrieval_docs} retrieval + {final_web_docs} web")
        if not self.config.agentic_search_mode:
            logger.info(f"[Retrieve Context] ├─ Per-query web search: {'Used by at least one query' if any_web_search_used else 'Not used by any query'}")
            logger.info(f"[Retrieve Context] ├─ Web search : {'YES' if web_search_needed else 'NO'} (based on per-query usage)")
        else:
            logger.info(f"[Retrieve Context] ├─ Final web search threshold: < 2 total documents")
            logger.info(f"[Retrieve Context] ├─ Final web search needed: {'YES' if web_search_needed else 'NO'}")
        if web_search_needed and len(docs) < 2:
            logger.info(f"[Retrieve Context] ├─ 🌐 Will perform additional web search with rewritten query")
        logger.info(f"[Retrieve Context] └─ 🏁 RETRIEVAL PHASE COMPLETED")
        logger.info("\n" + "=" * 78)
        
        return {
            "context_docs": docs,
            "requires_web_search": web_search_needed,
        }
def create_conversation_graph(
    memory, organization_id=None, conversation_id=None
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    logger.info(f"[Conversation Graph Creation] Creating conversation graph for conversation: {conversation_id}")
    builder = GraphBuilder(
        organization_id=organization_id, conversation_id=conversation_id
    )
    return builder.build(memory)
