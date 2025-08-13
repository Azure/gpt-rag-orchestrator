import logging
import sys
from typing import List, Any, Optional
from dotenv import load_dotenv
load_dotenv()

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


def truncate_chat_history(
    chat_history: List[Any], max_messages: int = 6
) -> List[Any]:
    """
    Truncate chat history to the most recent messages.

    Args:
        chat_history: List of chat message objects (AIMessage, HumanMessage, or dict)
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
        logger.info(
            f"[Chat History Cleaning] Truncated to last {max_messages} messages"
        )
        return truncated_history
    else:
        logger.info(
            f"[Chat History Cleaning] Less than {max_messages} messages, no truncation needed"
        )
        return chat_history


def clean_chat_history_for_llm(chat_history: List[Any]) -> str:
    """
    Clean and format chat history for LLM consumption as a string.

    Args:
        chat_history: List of chat message objects (AIMessage, HumanMessage, or dict)

    Returns:
        Formatted chat history string in the format:
            Human: {message}
            AI Message: {message}
    """
    truncated_history = truncate_chat_history(chat_history)
    if not truncated_history:
        return ""

    formatted_history = []
    for message in truncated_history:
        if hasattr(message, 'content'):
            content = message.content
            if not content:
                continue
            
            if hasattr(message, 'type'):
                if message.type == 'human':
                    display_role = "Human"
                elif message.type == 'ai':
                    display_role = "AI Message"
                else:
                    display_role = "AI Message" 
            else:
                display_role = "AI Message"
                
        elif isinstance(message, dict):
            if not message.get("content"):
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role and content:
                display_role = "Human" if role == "user" else "AI Message"
            else:
                continue
        else:
            continue

        if content:
            formatted_history.append(f"{display_role}: {content}")

    logger.info(
        f"[Chat History Cleaning] Formatted {len(formatted_history)} messages for LLM consumption"
    )
    return "\n\n".join(formatted_history)


def extract_thread_id_from_history(conversation_history: List[dict]) -> Optional[str]:
    """
    Extract the most recent code_thread_id from conversation history.
    
    Args:
        conversation_history: List of conversation messages from the database
        
    Returns:
        Most recent thread_id if found, None otherwise
    """
    if not conversation_history:
        logger.debug("[Thread ID Extraction] No conversation history provided")
        return None
    
    logger.info(f"[Thread ID Extraction] Searching {len(conversation_history)} messages for thread_id")
    
    # Search backwards through history to find the most recent thread_id
    for message in reversed(conversation_history):
        if isinstance(message, dict) and message.get("role") == "assistant":
            thread_id = message.get("code_thread_id")
            if thread_id:
                logger.info(f"[Thread ID Extraction] Found thread_id in conversation history: {thread_id}")
                return thread_id
    
    logger.debug("[Thread ID Extraction] No thread_id found in conversation history")
    return None


def extract_last_mcp_tool_from_history(conversation_history: List[dict]) -> str:
    """
    Extract the most recent MCP tool used from conversation history.
    
    Args:
        conversation_history: List of conversation messages from the database
        
    Returns:
        Name of the last MCP tool used, empty string if none found
    """
    if not conversation_history:
        logger.debug("[MCP Tool Extraction] No conversation history provided")
        return ""
    
    logger.info(f"[MCP Tool Extraction] Searching {len(conversation_history)} messages for last MCP tool")
    
    for message in reversed(conversation_history):
        if isinstance(message, dict) and message.get("role") == "assistant":
            last_mcp_tool = message.get("last_mcp_tool_used")
            if last_mcp_tool:
                logger.info(f"[MCP Tool Extraction] Found last MCP tool in conversation history: {last_mcp_tool}")
                return last_mcp_tool
    
    logger.debug("[MCP Tool Extraction] No MCP tool usage found in conversation history")
    return ""