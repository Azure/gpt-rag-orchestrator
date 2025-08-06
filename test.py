import os
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional

load_dotenv()

# Configuration constants
DEFAULT_ORGANIZATION_ID = "6c33b530-22f6-49ca-831b-25d587056237"
DEFAULT_RERANKER_THRESHOLD = 2.0
DEFAULT_CHAT_HISTORY = "chat history is not available"
SYSTEM_PROMPT = "You are a helpful assistant that helps determine the tools to use to answer the user's question. As of right now, you should only use agentic_search tool to answer the user's question."
REWRITTEN_QUERY = "Definition of consumer segmentation"

client = MultiServerMCPClient(
    {
        "search": {
            "url": "https://mcp-server-0v0r.onrender.com/mcp",
            "transport": "streamable_http",
        }
    }
)

llm = AzureChatOpenAI(
    temperature=0.4,
    openai_api_version="2025-04-01-preview",
    azure_deployment="gpt-4.1",
    streaming=False,
    timeout=30,
    max_retries=3,
    azure_endpoint=os.getenv("O1_ENDPOINT"),
    api_key=os.getenv("O1_KEY"),
)


def configure_agentic_search_args(
    tool_call: Dict[str, Any], 
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    rewritten_query: str = REWRITTEN_QUERY,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
    chat_history: str = DEFAULT_CHAT_HISTORY
) -> Dict[str, Any]:
    """
    Configure additional arguments for agentic_search tool calls.
    
    Args:
        tool_call: The original tool call dictionary
        organization_id: Organization identifier for the search
        rewritten_query: The rewritten/processed version of the query
        reranker_threshold: Threshold for reranking search results
        chat_history: Historical conversation context
    
    Returns:
        Updated tool call arguments
    """
    if tool_call['name'] == "agentic_search":
        tool_call['args'].update({
            'organization_id': organization_id,
            'rewritten_query': rewritten_query,
            'reranker_threshold': reranker_threshold,
            'historical_conversation': chat_history
        })
        print(f"  Configured agentic_search with args: {tool_call['args']}")
    
    return tool_call['args']


def find_tool_by_name(tools: List[Any], tool_name: str) -> Optional[Any]:
    """
    Find a tool in the tools list by its name.
    
    Args:
        tools: List of available tools
        tool_name: Name of the tool to find
    
    Returns:
        The tool object if found, None otherwise
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]], 
    tools: List[Any],
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    rewritten_query: str = REWRITTEN_QUERY,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
    chat_history: str = DEFAULT_CHAT_HISTORY
) -> List[Any]:
    """
    Execute a list of tool calls and return their results.
    
    Args:
        tool_calls: List of tool calls to execute
        tools: List of available tools
        organization_id: Organization identifier
        rewritten_query: The rewritten query for agentic_search
    
    Returns:
        List of tool execution results
    """
    tool_results = []
    
    if not tool_calls:
        print("  No tool calls to execute")
        return tool_results
    
    print(f"  Executing {len(tool_calls)} tool(s)...")
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        
        # Configure tool arguments based on tool type
        configure_agentic_search_args(
            tool_call, 
            organization_id=organization_id,
            rewritten_query=rewritten_query,
            reranker_threshold=reranker_threshold,
            chat_history=chat_history
        )
        
        # Find and execute the tool
        tool = find_tool_by_name(tools, tool_name)
        if tool:
            try:
                print(f"  Running {tool_name}...")
                tool_result = await tool.ainvoke(tool_call['args'])
                tool_results.append(tool_result)
                print(f"  ✓ {tool_name} completed successfully")
            except Exception as e:
                print(f"  ✗ Error executing {tool_name}: {e}")
                tool_results.append(f"Error: {e}")
        else:
            error_msg = f"Tool '{tool_name}' not found in available tools"
            print(f"  ✗ {error_msg}")
            tool_results.append(error_msg)
    
    return tool_results


async def get_llm_tool_calls(query: str, tools: List[Any]) -> List[Dict[str, Any]]:
    """
    Get tool calls from the LLM based on the user query.
    
    Args:
        query: User's question/query
        tools: List of available tools
    
    Returns:
        List of tool calls suggested by the LLM
    """
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    response = await llm_with_tools.ainvoke(messages)
    
    print(f"📋 LLM selected {len(response.tool_calls)} tool(s)")
    for i, tool_call in enumerate(response.tool_calls, 1):
        print(f"   {i}. {tool_call['name']} with args: {tool_call['args']}")
    
    return response.tool_calls


def log_execution_summary(execution_time: float, tool_results: List[Any]) -> None:
    """
    Log a summary of the execution results.
    
    Args:
        execution_time: Time taken for execution in seconds
        tool_results: List of tool execution results
    """
    print(f"\n🏁 Execution Summary:")
    print(f"   ⏱️  Completed in {execution_time:.2f} seconds")
    print(f"   📊 Retrieved {len(tool_results)} result(s)")
    
    if tool_results:
        preview = str(tool_results[0])
        # Truncate long previews for readability
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"   👀 First result preview: {preview}")


async def main():
    """
    Main execution function that orchestrates the tool calling process.
    """
    print("🚀 Starting tool execution process...")
    
    # Configuration
    query = "What is consumer segmentation? I am opening a gym located in San Francisco and I would like to figure out the target audience for my gym."
    organization_id = DEFAULT_ORGANIZATION_ID
    rewritten_query = REWRITTEN_QUERY
    reranker_threshold = DEFAULT_RERANKER_THRESHOLD
    chat_history = DEFAULT_CHAT_HISTORY
    
    start_time = time.time()
    
    try:
        # Step 1: Get available tools
        print("🔧 Fetching available tools...")
        tools = await client.get_tools()
        print(f"   Found {len(tools)} available tool(s)")
        
        # Step 2: Get tool calls from LLM
        print("\n Getting tool recommendations from LLM...")
        tool_calls = await get_llm_tool_calls(query, tools)
        
        # Step 3: Execute the tools
        print(f"\n Executing tools...")
        tool_results = await execute_tool_calls(
            tool_calls, 
            tools, 
            organization_id=organization_id,
            rewritten_query=rewritten_query,
            reranker_threshold=reranker_threshold,
            chat_history=chat_history
        )
        
        # Step 4: Log execution summary
        execution_time = time.time() - start_time
        log_execution_summary(execution_time, tool_results)
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n Execution failed after {execution_time:.2f} seconds: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())