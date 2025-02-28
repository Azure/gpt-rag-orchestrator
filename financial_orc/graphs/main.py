import os
import json
import requests
from typing import List, Annotated, Sequence, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
)
from pydantic import BaseModel

# tools
from .tools.tavily_tool import (
    conduct_tavily_search_news,
    conduct_tavily_search_general,
    format_tavily_results,
)
from .tools.database_retriever import CustomRetriever, format_retrieved_content
from datetime import datetime


########################################
# Define agent graph
########################################


class AgentState(TypedDict):
    """The state of the agent."""
    
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str
    chat_summary: str = ""


class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2024-05-01-preview"
    azure_deployment: str = "gpt-4o-orchestrator"
    index_name: str = "financial-index"
    retriever_top_k: int = 1
    reranker_threshold: float = 1.2
    web_search_results: int = 2
    temperature: float = 0.3
    max_tokens: int = 5000
    verbose: bool = True


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
        documentName: str = None,
        verbose: bool = True,
    ):
        """Initialize with configuration and validate environment variables"""
        # Validate required environment variables
        required_env_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_AI_SEARCH_API_KEY",
            "AZURE_SEARCH_API_VERSION",
            "AZURE_SEARCH_SERVICE",
        ]

        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize instance variables
        self.organization_id = organization_id
        self.config = config
        self.retriever = self._init_retriever()
        self.conversation_id = conversation_id
        self.documentName = documentName

    def _init_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="Agent",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3,
        )
        
    def _should_continue(self) -> str:
        """Route query based on knowledge requirement."""
        # not sure of how to make the comparison here
        return "return_state"

    def _init_retriever(self) -> CustomRetriever:
        try:
            config = self.config
            index_name = self.config.index_name
            return CustomRetriever(
                indexes=[index_name],
                topK=self.config.retriever_top_k,
                reranker_threshold=self.config.reranker_threshold,
                organization_id=self.organization_id,
                verbose=self.config.verbose
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure AI Search Retriever: {str(e)}"
            )

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        # set up graph
        graph = StateGraph(AgentState)

        # Add processing nodes
        graph.add_node("report_preload", self._report_retriever)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._tool_node)
        graph.add_node("return_state", self._return_state)

        # Define graph flow
        graph.add_edge(START, "report_preload")
        graph.add_edge("report_preload", "agent")
        graph.add_conditional_edges(
            "agent",
            self._orchestrator,
            {"continue": "tools", "return_state": "return_state"},
        )
        graph.add_edge("tools", "return_state")
        graph.add_edge("return_state", END)

        # Compile the graph
        return graph.compile(checkpointer=memory)

    def _report_retriever(self, state: AgentState) -> dict:
        """Retrieve the initial report."""
        try:
            # Get documents from retriever
            documents = self.retriever.invoke(self.documentName)
            # format the retrieved content
            formatted_content = format_retrieved_content(documents)

            if self.config.verbose:
                print(
                    f"[financial-orchestrator-agent] RETRIEVED DOCUMENTS: {len(documents)}"
                )

            if not documents or len(documents) == 0:
                if self.config.verbose:
                    print(
                        "[financial-orchestrator-agent] No documents retrieved, using fallback content"
                    )
                documents = [
                    Document(page_content="No information found about the report")
                ]

            return {"report": formatted_content}
        except Exception as e:
            if self.config.verbose:
                print(
                    f"[financial-orchestrator-agent] Error in report_retriever: {str(e)}"
                )
            # Return a fallback document to prevent crashes
            return {
                "report": [
                    Document(page_content="Error retrieving report information.")
                ]
            }

    class ToolDecision(BaseModel):
        orc_decision: Literal["tools", "return_state"]

    def _agent(self, state: AgentState) -> dict:
        """Currently doing nothing, it just exists as a dummy node"""
        pass

    def _orchestrator(self, state: AgentState) -> dict:
        """Decide if the question should be answered using a tool or not."""

        system_prompt = """ 
        You're a helpful assistant. Please decide if the question should be answered using a tool or not.
        """
        user_query = state.get("question")
        #user_query = state.get("messages", ["no messages"])[-1].content
        prompt = f"""
        Question: {user_query}
        """

        structured_llm = self._init_llm().with_structured_output(self.ToolDecision)
        response = structured_llm.invoke(prompt)

        if response.orc_decision == "tools":
            return "tools"
        else:
            return "return_state"

    @tool
    def web_search(query: str) -> str:
        """Conduct web search for user query that is not included in the report
        This tool should be used when the question is about recent news or events."""

        # Step 2. Executing a context search query
        result = conduct_tavily_search_news(query)
        # format the results
        formatted_results = format_tavily_results(result)

        return formatted_results

    @tool
    def general_search(query: str) -> str:
        """Conduct general web search for user query that is not included in the report
        This tool should be used when the question is more general, and the requested information does not have to be up to date.
        """
        result = conduct_tavily_search_general(query)
        formatted_results = format_tavily_results(result)
        return formatted_results

    tools = [web_search, general_search]

    # create a dictionary of tools by name
    tools_by_name = {tool.name: tool for tool in tools}

    def _tool_node(self, state: AgentState):
        outputs = []
        try:
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
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

    def _return_state(self, state: AgentState) -> dict:
        return {
            "messages": state["messages"],
            "report": state["report"],
            "chat_summary": state["chat_summary"],
        }


def create_conversation_graph(
    memory, organization_id=None, conversation_id=None, documentName=None
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    print(f"Creating conversation graph for organization: {organization_id}")
    builder = GraphBuilder(
        organization_id=organization_id, conversation_id=conversation_id, documentName=documentName
    )
    return builder.build(memory)

def format_chat_history(messages):
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


###################################################
# this function is to be modified and moved to the orchestrator.py file
###################################################


def call_model(state: AgentState, config: RunnableConfig):
    # check if state report if it exists, then use it, otherwise use the fallback content
    if state.get("report"):
        report = state["report"][0].page_content
        report_citation = state["report"][0].metadata.get("citation", "")
    else:
        report = "No report found, using websearch content"  # maybe redudant, but just in case we by pass the report retrieval step in the future
        report_citation = ""

    # Get the most recent message except the latest one
    chat_history = state["messages"][:-1] or []

    # Format chat history using the new formatting function
    formatted_chat_history = format_chat_history(
        chat_history
    )  # this function has already been moved to the orchestrator.py file

    # Get chat summary
    chat_summary = state.get("chat_summary", "")

    # get curretn date
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = """
    You are a helpful assistant. Today's date is {current_date}.
    Use available tools to answer queries if provided information is irrelevant. You should only use sources within the past 6 months.
    Consider conversation history for context in your responses if available. 
    If the context is already relevant, then do not use any tools.
    Treat the report as a primary source of information, **prioritize it over the tool call results**.
    
    ***Important***: 
    - If the tool is triggered, then mention in the response that external sources were used to supplement the information. You must also provide the URL of the source in the response.
    - Do not use your pretrained knowledge to answer the question.
    - YOU MUST INCLUDE CITATIONS IN YOUR RESPONSE FOR EITHER THE REPORT OR THE WEB SEARCH RESULTS. You will be penalized $10,000 if you fail to do so. Here is an example of how you should format the citation:
    - Citation format: [[1]](https://www.example.com)

    Citation Example:
    ```
    Renewable energy sources, such as solar and wind, are significantly more efficient and environmentally friendly compared to fossil fuels. Solar panels, for instance, have achieved efficiencies of up to 22% in converting sunlight into electricity [[1]](https://renewableenergy.org/article8.pdf?s=solarefficiency&category=energy&sort=asc&page=1). 
    These sources emit little to no greenhouse gases or pollutants during operation, contributing far less to climate change and air pollution [[2]](https://environmentstudy.com/article9.html?s=windenergy&category=impact&sort=asc). In contrast, fossil fuels are major contributors to air pollution and greenhouse gas emissions, which significantly impact human health and the environment [[3]](https://climatefacts.com/article10.csv?s=fossilfuels&category=emissions&sort=asc&page=3).
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
        current_date=current_date,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

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
    # def conv_length_check(state: AgentState):
    #     return state

    # # define the conditional edge that determines whether to continue or not
    # def chat_length_check(state: AgentState) -> Literal['summarize_chat', "__end__"]:
    #     """summarize the conversation if msg > 6, if not then end """
    #     # Filter out ToolMessages and count only user/assistant messages
    #     message_count = len([
    #         msg for msg in state['messages']
    #         if not isinstance(msg, ToolMessage)
    #     ])

    #     if verbose:
    #         print(f"[financial-orchestrator-agent] Message count: {message_count}")

    #     if message_count > 6:
    #         return "summarize_chat"
    #     return "__end__"

    # def summarize_chat(state: AgentState):
    #     from langchain_core.messages import HumanMessage, RemoveMessage

    #     # Get existing summary
    #     summary = state.get("chat_summary", "")

    #     # Filter out tool messages for summarization
    #     messages_to_summarize = [
    #         msg for msg in state['messages']
    #         if not isinstance(msg, ToolMessage)
    #     ]

    #     if verbose:
    #         print(f"[financial-orchestrator-agent] Summarizing {len(messages_to_summarize)} messages")

    #     # Create summary prompt
    #     if summary:
    #         summary_msg = (
    #             f"This is the summary of the conversation so far: \n\n{summary}\n\n"
    #             "Extend the summary by taking into account the new messages above. "
    #             "Just return the summary, no need to say anything else:"
    #         )
    #     else:
    #         summary_msg = (
    #             "Summarize the following conversation history. "
    #             "Just return the summary, no need to say anything else:"
    #         )

    #     # Create messages for summary generation
    #     final_summary_msg = messages_to_summarize + [HumanMessage(content=summary_msg)]

    #     # Generate summary
    #     new_summary = llm.invoke(final_summary_msg)
    # ###################################################
    # ### will need to revisit this later
    # ###################################################
    #     # # Keep only the last two non-tool messages
    #     # messages_to_keep = [
    #     #     msg for msg in state['messages'][-4:]
    #     #     if not isinstance(msg, ToolMessage)
    #     # ][-2:]

    #     # # Create remove messages for all except the kept ones
    #     # messages_to_remove = [
    #     #     RemoveMessage(id=msg.id)
    #     #     for msg in state['messages']
    #     #     if msg not in messages_to_keep
    #     # ]

    #     # Store in CosmosDB
    #     conversation_id = state.get('configurable', {}).get('thread_id', '')
    #     if conversation_id:
    #         try:
    #             conversation_data = get_conversation_data(conversation_id)
    #             if conversation_data:
    #                 conversation_data['summary'] = new_summary.content
    #                 update_conversation_data(conversation_id, conversation_data)
    #                 if verbose:
    #                     print(f"[financial-orchestrator-agent] Updated summary in CosmosDB for conversation {conversation_id}")
    #         except Exception as e:
    #             if verbose:
    #                 print(f"[financial-orchestrator-agent] Failed to update summary in CosmosDB: {str(e)}")

    #     return {
    #         "chat_summary": new_summary.content,
    #         # "messages": messages_to_remove
    #     }
