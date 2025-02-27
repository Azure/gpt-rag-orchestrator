import os
import uuid
import logging
import base64
import time
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from langchain_core.messages import AIMessage, ToolMessage
from .graphs.main import create_conversation_graph
from dataclasses import dataclass, field
from typing import List
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from typing import List, Annotated, Sequence, TypedDict, Literal
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field
from typing import Literal

from shared.prompts import (
    FINANCIAL_ANSWER_PROMPT,
    QUERY_REWRITING_PROMPT,
)
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG").upper())

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
###################################################
# Generation code 
###################################################

class ReportType(BaseModel):
    """Categorize user query into one of the predefined report types.
    
    Attributes:
        report_type: The classified type of report based on user query
    """
    report_type: Literal[
        "monthly_economics",
        "weekly_economics", 
        "company_analysis",
        "home_improvement",
        "ecommerce",
        "creative_brief"
    ] = Field(
        default="monthly_economics",
        description="Report classification based on query content",
        title="Report Type Classification"
    )

@dataclass
class AgentState:
    """The state of the agent."""
    
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str
    chat_summary: str = ""
class FinancialOrchestrator:
    """Manages conversation flow and state between user and AI agent.
    
    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
    """
    
    def __init__(self, organization_id: str = None):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self.organization_id = organization_id
        
    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        serialized = memory.serde.dumps(memory.get_tuple(config))
        return base64.b64encode(serialized).decode("utf-8")
    
    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Decode and load conversation memory from base64 string."""
        memory = MemorySaver()
        if memory_data != "":
            decoded_data = base64.b64decode(memory_data)
            json_data = memory.serde.loads(decoded_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )
        return memory
    
    def categorize_query(query: str, llm: AzureChatOpenAI) -> str:
        """Categorize user query into predefined report types.
        
        Args:
            query: User's input query
            llm: Configured language model
            
        Returns:
            str: Classified report type
        """
        categorizer = llm.with_structured_output(ReportType)
        result = categorizer.invoke(query)
        return result.report_type
    
    def _clean_chat_history(self, chat_history: List[dict]) -> str:
        """
        Clean the chat history and format it as a string for LLM consumption.

        Args:
            chat_history (list): List of chat message dictionaries

        Returns:
            str: Formatted chat history string in the format:
                 Human: {message}
                 AI: {message}
        """
        formatted_history = []

        for message in chat_history:
            if not message.get("content"):
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role and content:
                display_role = "Human" if role == "user" else "AI Message"
                formatted_history.append(f"{display_role}: {content}")

        return "\n\n".join(formatted_history)
    
    def generate_response(
        self, conversation_id: str, question: str, user_info: dict, documentName: str
    ):
        """
        Process a conversation turn with the AI agent.

        Args:
            conversation_id: Unique identifier for conversation
            question: User's input question
            user_info: Dictionary containing user metadata

        Returns:
            dict: Response containing conversation_id, answer and thoughts
        """
        start_time = time.time()
        logging.info(f"[financial-orc] Gathering resources for: {question}")
        conversation_id = conversation_id or str(uuid.uuid4())
        
        response = {
            "content": "",
        }
        state = None

        try:
            # Load conversation state
            logging.info(f"[financial-orc] Loading conversation data")
            conversation_data = get_conversation_data(conversation_id)
            logging.info(f"[financial-orc] Loading memory")
            memory = self._load_memory(conversation_data.get("memory_data", ""))
            logging.info(f"[financial-orc] Memory loaded")
            # Process through agent

            # insert conversation to the memory object
            agent = create_conversation_graph(memory = memory, organization_id = self.organization_id, conversation_id = conversation_id, documentName = documentName)
            logging.info(f"[financial-orc] Agent created")
            config = {"configurable": {"thread_id": conversation_id}}

            
            # Get agent response
            logging.info(f"[financial-orc] Invoking agent")
            resources = agent.invoke({"question": question}, config)
            logging.error(f"[financial-orc] Agent resources")
            state = AgentState(
                question,
                resources.get("messages"),
                resources.get("report"),
                resources.get("chat_summary")
            )
        except Exception as e:
            logging.error(f"[financial-orc] Error retrieving resources: {str(e)}")
            yield "I'm sorry, I'm having trouble generating a response right now. Please try again later. Error: " + str(e)
            store_agent_error(user_info["id"], str(e), question)
            
        try:
            context = ""
            max_tokens = 2000
            if state.report:
                context = state.report
            response_llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o-orchestrator",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3
            )
            
            history = conversation_data.get("history", [])
            
            system_prompt = FINANCIAL_ANSWER_PROMPT
            prompt = f"""
            
            Question: 
            
            <----------- USER QUESTION ------------>

            ORIGINAL QUESTION: {state.question}

            <----------- END OF USER QUESTION ------------>
            
            
            Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)
            
            <----------- PROVIDED CONTEXT ------------>
            {context}
            <----------- END OF PROVIDED CONTEXT ------------>

            Chat History:

            <----------- PROVIDED CHAT HISTORY ------------>
            {self._clean_chat_history(history)}
            <----------- END OF PROVIDED CHAT HISTORY ------------>

            Chat Summary:

            <----------- PROVIDED CHAT SUMMARY ------------>
            {state.chat_summary}
            <----------- END OF PROVIDED CHAT SUMMARY ------------>

            Provide a detailed answer.
            """
            tokens = response_llm.stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            )
            try:
                while True:
                    try:
                        token = next(tokens)
                        if token:
                            response["content"] += f"{token.content}"
                            yield f"{token.content}"
                    except StopIteration:
                        break
            except Exception as e:
                logging.error(f"[orchestrator] Error generating response: {str(e)}")
                yield "I'm sorry, I'm having trouble generating a response right now. Please try again later. Error: " + str(e)
                store_agent_error(user_info["id"], str(e), question)
            logging.info(f"[orchestrator] Response generated: {response['content']}")
        except Exception as e:
            logging.error(f"[financial-orc] Error generating response: {str(e)}")
            yield "I'm sorry, I'm having trouble generating a response right now. Please try again later. Error: " + str(e)
            store_agent_error(user_info["id"], str(e), question)
        # Serialize and store updated memory
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        b64_memory = base64.b64encode(serialized_data).decode("utf-8")

        # set values on cosmos object
        conversation_data["memory_data"] = b64_memory

        # Add new messages to history
        conversation_data["history"].append({"role": "user", "content": question})
        conversation_data["history"].append(
            {"role": "assistant", "content": response["content"]}
        )

        # conversation data
        response_time = round(time.time() - start_time, 2)
        interaction = {
            "user_id": user_info["id"],
            "user_name": user_info["name"],
            "response_time": response_time,
        }
        conversation_data["interaction"] = interaction

        # Store in CosmosDB
        update_conversation_data(conversation_id, conversation_data)

def run(conversation_id, question, documentName, client_principal):
    try:
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())

        logging.info(
            f"[financial-orchestrator] Initiating conversation with id: {conversation_id}"
        )

        # Get existing conversation data from CosmosDB
        logging.info("[financial-orchestrator] Loading conversation data")
        conversation_data = get_conversation_data(conversation_id, type="financial")

        start_time = time.time()

        # Initialize memory with existing data
        memory = MemorySaver()
        if conversation_data and conversation_data.get("memory_data"):
            memory_data = base64.b64decode(conversation_data["memory_data"])
            json_data = memory.serde.loads(memory_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )

        end_time = time.time()
        logging.info(
            f"[financial-orchestrator] Conversation data loaded in {end_time - start_time} seconds"
        )

        # Create and invoke agent
        agent_executor = create_conversation_graph(
            memory=memory, documentName=documentName
        )
        config = {"configurable": {"thread_id": conversation_id}}

        # init an LLM since we don't have one 
        response_llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o-orchestrator",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3,
        )
        
        response = ""

        tokens = response_llm.stream(question)
        try:
            while True:
                try:
                    token = next(tokens)
                    if token:
                        response["content"] += f"{token.content}"
                        yield f"{token.content}"
                except StopIteration:
                    break
        except Exception as e:
            logging.error(f"[orchestrator] Error generating response: {str(e)}")
            response["content"] = (
                "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            )
            yield response["content"]
        logging.info(f"[orchestrator] Response generated: {response['content']}")

        # Serialize and store updated memory
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        b64_memory = base64.b64encode(serialized_data).decode("utf-8")

        # set values on cosmos object
        conversation_data["memory_data"] = b64_memory

        # Add new messages to history
        conversation_data["history"].append({"role": "user", "content": question})
        conversation_data["history"].append(
            {"role": "assistant", "content": response.content}
        )

        # conversation data
        response_time = round(time.time() - start_time, 2)
        interaction = {
            "user_id": client_principal["id"],
            "user_name": client_principal["name"],
            "response_time": response_time,
        }
        conversation_data["interaction"] = interaction

        # Store in CosmosDB
        update_conversation_data(conversation_id, conversation_data)

        yield {
            "conversation_id": conversation_id,
            "answer": response.content,
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
    except Exception as e:
        logging.error(f"[financial-orchestrator] {conversation_id} error: {str(e)}")
        store_agent_error(client_principal["id"], str(e), question)
        yield {
            "conversation_id": conversation_id,
            "answer": f"Error processing request: {str(e)}",
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
