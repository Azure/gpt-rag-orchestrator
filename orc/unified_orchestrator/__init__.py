"""
Unified Conversation Orchestrator Module

This module provides a streamlined orchestrator for conversation processing
using LangGraph workflows and MCP tool integration.

The orchestrator coordinates multiple components to process user queries:
- StateManager: Manages conversation persistence with Cosmos DB
- ContextBuilder: Builds context from organization data and history
- QueryPlanner: Handles query rewriting, augmentation, and categorization
- MCPClient: Manages MCP server connections and tool execution
- ResponseGenerator: Generates streaming LLM responses

Public API:
- ConversationOrchestrator: Main entry point for conversation processing
- ConversationState: State object that flows through the workflow
- OrchestratorConfig: Configuration dataclass for customizing behavior

Example usage:
    from orc.unified_orchestrator import ConversationOrchestrator
    
    orchestrator = ConversationOrchestrator(organization_id="org123")
    async for item in orchestrator.generate_response_with_progress(
        conversation_id="conv456",
        question="What is our brand positioning?",
        user_info={"id": "user789"}
    ):
        # Process streaming response
        print(item)
"""

from .orchestrator import ConversationOrchestrator
from .models import ConversationState, OrchestratorConfig
from .enums import VerbosityLevel, VERBOSITY_PROMPTS

__all__ = [
    "ConversationOrchestrator",
    "ConversationState",
    "OrchestratorConfig",
    "VerbosityLevel",
    "VERBOSITY_PROMPTS",
]
