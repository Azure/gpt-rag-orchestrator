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
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
)


# Define agent graph
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str
    chat_summary: str = ""


# Update CustomRetriever with error handling
class CustomRetriever(BaseRetriever):
    """
    Custom retriever class that extends BaseRetriever to work with Azure AI Search.

    Attributes:
        topK (int): Number of top results to retrieve.
        reranker_threshold (float): Threshold for reranker score.
        indexes (List): List of index names to search.
    """

    topK = 1
    reranker_threshold = 1.2 
    vector_similarity_threshold = 0.5
    semantic_config = "financial-index-semantic-configuration"
    index_name = "financial-index"
    indexes: List
    verbose: bool

    def get_search_results(
        self,
        query: str,
        indexes: list,
        k: int = topK,
        semantic_config: str = semantic_config,
        reranker_threshold: float = reranker_threshold,  # range between 0 and 
        vector_similarity_threshold: float = vector_similarity_threshold,
    ) -> List[dict]:
        """
        Performs multi-index hybrid search and returns ordered dictionary with the combined results.

        Args:
            query (str): The search query.
            indexes (list): List of index names to search.
            k (int): Number of top results to retrieve. Default is 5.
            reranker_threshold (float): Threshold for reranker score. Default is 1.2.

        Returns:
            OrderedDict: Ordered dictionary of search results.
        """

        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ["AZURE_AI_SEARCH_API_KEY"],
        }
        params = {"api-version": os.environ["AZURE_SEARCH_API_VERSION"]}

        agg_search_results = dict()

        for index in indexes:
            search_payload = {
                "search": query,
                "select": "chunk_id, file_name, chunk, url, date_last_modified",
                "queryType": "semantic",
                "vectorQueries": [
                    {
                        "text": query,
                        "fields": "text_vector",
                        "kind": "text",
                        "k": k,
                        "threshold": {
                            "kind": "vectorSimilarity",
                            "value": vector_similarity_threshold,  # 0.333 - 1.00 (Cosine), 0 to 1 for Euclidean and DotProduct.
                        },
                    }
                ],
                "semanticConfiguration": semantic_config,  # change the name depends on your config name
                "captions": "extractive",
                "answers": "extractive",
                "count": "true",
                "top": k,
            }

            try:
                resp = requests.post(
                    os.environ["AZURE_SEARCH_ENDPOINT"]
                    + "/indexes/"
                    + index
                    + "/docs/search",
                    data=json.dumps(search_payload),
                    headers=headers,
                    params=params,
                )
                # resp.raise_for_status()
                search_results = resp.json()
                agg_search_results[index] = search_results
            except Exception as e:
                if self.verbose:
                    print(f"[financial-orchestrator-agent] Error in get_search_results: {str(e)}")
                return []

        content = dict()
        ordered_content = OrderedDict()

        for index, search_results in agg_search_results.items():
            for result in search_results["value"]:
                if (
                    result["@search.rerankerScore"] > reranker_threshold
                ):  # Range between 0 and 4
                    content[result["chunk_id"]] = {
                        "filename": result["file_name"],
                        "chunk": (result["chunk"] if "chunk" in result else ""),
                        "location": (result["url"] if "url" in result else ""),
                        "date_last_modified": result["date_last_modified"],
                        "caption": result["@search.captions"][0]["text"],
                        "score": result["@search.rerankerScore"],
                        "index": index,
                    }

        topk = k

        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding topK results
                break

        return ordered_content

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents based on the given query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: List of relevant documents.
        """

        ordered_results = self.get_search_results(
            query,
            self.indexes,
            k=self.topK,
            reranker_threshold=self.reranker_threshold,
        )
        top_docs = []

        for key, value in ordered_results.items():
            location = value["location"] if value["location"] is not None else ""
            top_docs.append(
                Document(
                    page_content=value["chunk"],
                    metadata={"source": location, "score": value["score"]},
                )
            )

        return top_docs


def create_main_agent(checkpointer, documentName, verbose=True):
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

            return {"report": documents}
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

        search = TavilySearchAPIRetriever(
            k=2,
            search_depth="advanced",
            include_generated_answer=True,
            include_raw_content=True,
        )

        # Convert the search results to a string representation
        results = search.invoke(query)

        # Format the results into a readable string
        formatted_results = []
        for result in results:
            metadata = result.metadata
            page_content = result.page_content
            formatted_results.append(
                f"URL: {metadata['source']}\n" f"Content: {page_content}\n"
            )

        if verbose:
            print(f"[financial-orchestrator-agent] WEBSEARCH RESULTS: {len(results)}")

        return "\n\n".join(formatted_results)

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
                    formatted_messages.append(f"Output: {msg.content[:200]}")
                except:
                    formatted_messages.append(f"{msg.content[:200]}")
        
        # Add final separator
        formatted_messages.append("-" * 50)
        
        # Join all lines with newlines
        return "\n".join(formatted_messages)

    def call_model(state: AgentState, config: RunnableConfig):
        report = state["report"][0].page_content
        
        # Get the most recent message except the latest one 
        chat_history = state['messages'][:-1] or []

        # Format chat history using the new formatting function
        formatted_chat_history = format_chat_history(chat_history)

        # Get chat summary
        chat_summary = state.get("chat_summary", "")

        system_prompt = """
        You are a helpful assistant. Use available tools to answer queries if provided information is irrelevant. 
        Consider conversation history for context in your responses if available. 
        If the context is already relevant, then do not use any tools.
        Treat the report as a primary source of information, but use the web search tool to supplement the information if needed.
        
        ***Important***: If the tool is triggered, then mention in the response that external sources were used to supplement the information. You must also provide the URL of the source in the response.
        
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
            chat_summary=chat_summary
        )

        # if verbose:                
        #     print(f"[financial-orchestrator-agent] Formatted system prompt:\n{system_prompt}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
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
                print("[financial-orchestrator-agent] No tool calls, checking conversation length")
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
