import os
import json
import requests
from collections import OrderedDict
from typing import List, Annotated, Sequence, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
)

# tools
from .tools.tavily_tool import conduct_tavily_search, format_tavily_results
from .tools.database_retriever import CustomRetriever, format_retrieved_content
from datetime import datetime


########################################
# Define agent graph
########################################

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str
    chat_summary: str = ""

    
def create_main_agent(checkpointer, documentName, verbose=True):

    # validate env variables
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_AI_SEARCH_API_KEY",
        "AZURE_SEARCH_API_VERSION",
        "AZURE_SEARCH_SERVICE"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    # Define model
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="Agent",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.3,
    )
    ###################################################
    # Define retriever tool
    ###################################################
    # Define the index name for Azure AI Search
    index_name = "financial-index"
    indexes = [index_name]  # we can add more indexes here if needed

    # Initialize custom retriever on Azure AI Search
    k = 1
    retriever = CustomRetriever(
        indexes=indexes,
        topK=k,
        reranker_threshold=1.2,
        verbose=verbose,
    )

    def report_retriever(state: AgentState):
        """Retrieve the initial report."""
        try:
            # Get documents from retriever
            documents = retriever.invoke(documentName)
            # format the retrieved content
            formatted_content = format_retrieved_content(documents)
            
            if verbose:
                print(f"[financial-orchestrator-agent] RETRIEVED DOCUMENTS: {len(documents)}")
                # print(f"[financial-orchestrator-agent] DOCUMENT NAME: {documentName}")
                # print(documents[0].page_content)

            if not documents or len(documents) == 0:
                if verbose:
                    print("[financial-orchestrator-agent] No documents retrieved, using fallback content")
                documents = [
                    Document(
                        page_content="No information found about the report"
                    )
                ]

            return {"report": formatted_content}
        except Exception as e:
            if verbose:
                print(f"[financial-orchestrator-agent] Error in report_retriever: {str(e)}")
            # Return a fallback document to prevent crashes
            return {
                "report": [
                    Document(
                        page_content="Error retrieving report information."
                    )
                ]
            }
    ###################################################
    # Define web search tool
    ###################################################
    @tool
    def web_search(query: str) -> str:
        """Conduct web search for user query that is not included in the report"""

        # Step 2. Executing a context search query
        result = conduct_tavily_search(query)
        # format the results
        formatted_results = format_tavily_results(result)

        return formatted_results
    

    tools = [web_search]

    ###################################################
    # define nodes and edges
    ###################################################
    # Tool dictionary definition remains the same
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: AgentState):
        outputs = []
        try:
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        except KeyError as e:
            raise ValueError(f"Tool not found: {e}")
        except Exception as e:
            raise Exception(f"Error processing tool calls: {e}")
        return {"messages": outputs}
    

    def format_chat_history(messages):
        from langchain_core.messages import HumanMessage
        """Format chat history into a clean, readable string."""
        if not messages:
            return "No previous conversation history."
            
        formatted_messages = []
        for msg in messages:
            # Add a separator line
            formatted_messages.append("-" * 50)
            
            # Format based on message type
            if isinstance(msg, HumanMessage):
                formatted_messages.append("Human:")
                formatted_messages.append(f"{msg.content}")
                
            elif isinstance(msg, AIMessage):
                formatted_messages.append("Assistant:")
                formatted_messages.append(f"{msg.content}")
                
            elif isinstance(msg, ToolMessage):
                formatted_messages.append("Tool Output:")
                # Try to format tool output nicely
                try:
                    tool_name = getattr(msg, 'name', 'Unknown Tool')
                    formatted_messages.append(f"Tool: {tool_name}")
                    formatted_messages.append(f"Output: {msg.content}")
                except:
                    formatted_messages.append(f"{msg.content}")
        
        # Add final separator
        formatted_messages.append("-" * 50)
        
        # Join all lines with newlines
        return "\n".join(formatted_messages)

    def call_model(state: AgentState, config: RunnableConfig):
        # check if state report if it exists, then use it, otherwise use the fallback content
        if state.get("report"):
            report = state["report"][0].page_content
            report_citation = state["report"][0].metadata.get("citation", "")
        else:
            report = "No report found, using websearch content" # maybe redudant, but just in case we by pass the report retrieval step in the future
            report_citation = ""
            
        # Get the most recent message except the latest one 
        chat_history = state['messages'][:-1] or []

        # Format chat history using the new formatting function
        formatted_chat_history = format_chat_history(chat_history)

        # Get chat summary
        chat_summary = state.get("chat_summary", "")

        # get curretn date 
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_prompt = """
        You are a helpful assistant. Today's date is {current_date}.
        Use available tools to answer queries if provided information is irrelevant. You should only use sources within the past 6 months.
        Consider conversation history for context in your responses if available. 
        If the context is already relevant, then do not use any tools.
        Treat the report as a primary source of information, prioritize it over the web search results.
        
        ***Important***: 
        - If the tool is triggered, then mention in the response that external sources were used to supplement the information. You must also provide the URL of the source in the response.
        - Do not use your pretrained knowledge to answer the question.
        - YOU MUST INCLUDE CITATIONS IN YOUR RESPONSE FOR EITHER THE REPORT OR THE WEB SEARCH RESULTS. You will be penalized $10,000 if you fail to do so. Here is an example of how you should format the citation:

        Citation Example:
        ```
        Renewable energy sources, such as solar and wind, are significantly more efficient and environmentally friendly compared to fossil fuels. Solar panels, for instance, have achieved efficiencies of up to 22% in converting sunlight into electricity [[1]](https://renewableenergy.org/article8.pdf?s=solarefficiency&category=energy&sort=asc&page=1). These sources emit little to no greenhouse gases or pollutants during operation, contributing far less to climate change and air pollution [[2]](https://environmentstudy.com/article9.html?s=windenergy&category=impact&sort=asc). In contrast, fossil fuels are major contributors to air pollution and greenhouse gas emissions, which significantly impact human health and the environment [[3]](https://climatefacts.com/article10.csv?s=fossilfuels&category=emissions&sort=asc&page=3).
        ```
        Report Citation:
    
        {report_citation}
        ==================
        Report Information:

        {report}
        ==================

        Previous Conversation:

        {formatted_chat_history}
        ====================

        Conversation Summary:

        {chat_summary}
        """.format(
            report=report,
            formatted_chat_history=formatted_chat_history,
            chat_summary=chat_summary,
            report_citation=report_citation,
            current_date=current_date
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}. Please prioritize the report over the web search results."),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = agent_executor.invoke(
            {
                "input": state["messages"][-1].content,
            }
        )

        return {"messages": [AIMessage(content=response["output"])]}

    ###################################################
    # summarize chat history 
    ###################################################
    # summarize conversation at the end if total messages > 6
    
    # dummy node for conversation history 
    def conv_length_check(state: AgentState):
        return state
    
    # define the conditional edge that determines whether to continue or not
    def chat_length_check(state: AgentState) -> Literal['summarize_chat', "__end__"]:
        """summarize the conversation if msg > 6, if not then end """
        # Filter out ToolMessages and count only user/assistant messages
        message_count = len([
            msg for msg in state['messages'] 
            if not isinstance(msg, ToolMessage)
        ])
        
        if verbose:
            print(f"[financial-orchestrator-agent] Message count: {message_count}")
        
        if message_count > 6:
            return "summarize_chat"
        return "__end__"

    def summarize_chat(state: AgentState):
        from langchain_core.messages import HumanMessage, RemoveMessage
        
        # Get existing summary
        summary = state.get("chat_summary", "")
        
        # Filter out tool messages for summarization
        messages_to_summarize = [
            msg for msg in state['messages'] 
            if not isinstance(msg, ToolMessage)
        ]
        
        if verbose:
            print(f"[financial-orchestrator-agent] Summarizing {len(messages_to_summarize)} messages")
        
        # Create summary prompt
        if summary:
            summary_msg = (
                f"This is the summary of the conversation so far: \n\n{summary}\n\n"
                "Extend the summary by taking into account the new messages above. "
                "Just return the summary, no need to say anything else:"
            )
        else:
            summary_msg = (
                "Summarize the following conversation history. "
                "Just return the summary, no need to say anything else:"
            )
        
        # Create messages for summary generation
        final_summary_msg = messages_to_summarize + [HumanMessage(content=summary_msg)]
        
        # Generate summary
        new_summary = llm.invoke(final_summary_msg)
    ###################################################
    ### will need to revisit this later
    ###################################################
        # # Keep only the last two non-tool messages
        # messages_to_keep = [
        #     msg for msg in state['messages'][-4:] 
        #     if not isinstance(msg, ToolMessage)
        # ][-2:]
        
        # # Create remove messages for all except the kept ones
        # messages_to_remove = [
        #     RemoveMessage(id=msg.id) 
        #     for msg in state['messages'] 
        #     if msg not in messages_to_keep
        # ]
        
        # Store in CosmosDB
        conversation_id = state.get('configurable', {}).get('thread_id', '')
        if conversation_id:
            try:
                conversation_data = get_conversation_data(conversation_id)
                if conversation_data:
                    conversation_data['summary'] = new_summary.content
                    update_conversation_data(conversation_id, conversation_data)
                    if verbose:
                        print(f"[financial-orchestrator-agent] Updated summary in CosmosDB for conversation {conversation_id}")
            except Exception as e:
                if verbose:
                    print(f"[financial-orchestrator-agent] Failed to update summary in CosmosDB: {str(e)}")
        
        return {
            "chat_summary": new_summary.content,
            # "messages": messages_to_remove
        }
        

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if verbose:
            print(f"[financial-orchestrator-agent] Checking continuation condition")
        
        # if there is no function call, then we check conversation length
        if not last_message.tool_calls:
            if verbose:
                print("[financial-orchestrator-agent] No tool calls in last message, checking conversation length")
            return "conv_length_check"
        
        # if there is a function call, we continue with tools
        if verbose:
            print("[financial-orchestrator-agent] Tool calls present, continuing with tools")
        return "continue"

    ###################################################
    # define graph
    ###################################################

    workflow = StateGraph(AgentState)

    # define the preload document node
    workflow.add_node("report_preload", report_retriever)
    # define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("conv_length_check", conv_length_check)
    workflow.add_node("summarize_chat", summarize_chat)

    # set the entry point as agent
    workflow.set_entry_point("report_preload")

    # we now add a conditional edge
    workflow.add_edge("report_preload", "agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "conv_length_check": "conv_length_check"}
    )

    # add a normal edge from tools to agent
    workflow.add_edge("tools", "agent")

    workflow.add_conditional_edges(
        "conv_length_check", chat_length_check, {"summarize_chat": "summarize_chat", "__end__": END}
    )
    workflow.add_edge("summarize_chat", END)

    # compile the graph
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
