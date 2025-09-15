from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain.schema import Document


@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_retrieval: Flag indicating if retrieval is needed
        rewritten_query: Rewritten query for better search
        query_category: Category of the query
        augmented_query: Augmented version of the query
        mcp_tool_used: List of MCP tools that were used
        tool_results: Results from tool execution
        code_thread_id: Thread id for the code interpreter tool
        last_mcp_tool_used: Name of the last MCP tool used
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(default_factory=list)
    context_docs: List[Document] = field(default_factory=list)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(default_factory=str)
    query_category: str = field(default_factory=str)
    augmented_query: str = field(default_factory=str)
    mcp_tool_used: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    code_thread_id: Optional[str] = field(default=None)
    last_mcp_tool_used: str = field(default="")
