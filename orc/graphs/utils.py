import logging
import sys
from typing import List, Any
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
    chat_history: List[Any], max_messages: int = 4
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