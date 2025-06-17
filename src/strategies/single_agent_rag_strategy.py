import logging
import os
from typing import Any, Optional

from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    AzureAISearchQueryType,
    AzureAISearchTool,
    BingGroundingTool,
    ListSortOrder,
    MessageDeltaChunk,
    MessageDeltaTextUrlCitationAnnotation,
    MessageTextContent,
    RunStep,
    ThreadMessage,
    ThreadRun,
)

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

# -----------------------------------------------------------------------------
# Be sure to configure the root logger at DEBUG level somewhere early in your app,
# e.g. in your main entrypoint:
#
#    logging.basicConfig(level=logging.DEBUG, 
#                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
#
# That way, all of the logging.debug(...) calls below will actually show up.
# -----------------------------------------------------------------------------

class SingleAgentRAGStrategy(BaseAgentStrategy):
    """
    Implements a single-agent Retrieval-Augmented Generation (RAG) strategy
    using Azure AI Foundry. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("Initializing SingleAgentRAGStrategy...")

        # Event handler for streaming responses
        self.strategy_type = AgentStrategies.SINGLE_AGENT_RAG
        self.event_handler = EventHandler()

        # Agent Tools Initialization Section
        # =========================================================
        self.tools_list = []
        self.tool_resources = {}

        # --- Initialize BingGroundingTool (if configured) ---
        bing_conn = self.cfg.get("BING_CONNECTION_ID")
        if not bing_conn:
            logging.warning(
                "BING_CONNECTION_ID not set in App Config variables. "
                "BingGroundingTool will not be available."
            )
        else:
            bing = BingGroundingTool(connection_id=bing_conn, count=5)
            bing_def = bing.definitions[0]
            self.tools_list.append(bing_def)
            logging.debug(f"Added BingGroundingTool to tools_list: {bing_def}")

        # --- Initialize AzureAISearchTool ---
        
        azure_ai_conn_id = self.cfg.get("SEARCH_CONNECTION_ID") 
        index_name = self.cfg.get("SEARCH_RAG_INDEX_NAME") 

        logging.debug(f"seachConnectionId (cfg)  = {azure_ai_conn_id}")
        logging.debug(f"SEARCH_RAG_INDEX_NAME (cfg) = {index_name}")
        if not azure_ai_conn_id:
            logging.warning(
                "seachConnectionId undefined (cfg). "
                "AzureAISearchTool will be unavailable."
            )
        if not index_name:
            logging.warning(
                "SEARCH_RAG_INDEX_NAME undefined (cfg). "
                "AzureAISearchTool will be unavailable."
            )

        # Create the AzureAISearchTool instance.
        # Note: if you already have index_asset_id, you can pass it here as well:
        #    AzureAISearchTool(index_connection_id=..., index_asset_id=..., index_name=..., ...)
        self.ai_search = AzureAISearchTool(
            index_connection_id=azure_ai_conn_id,
            index_name=index_name,
            query_type=AzureAISearchQueryType.SIMPLE, 
            top_k=3,
            filter="",  # Optional filter for search results
        )

        # Log out the definition object and resource keys so you can inspect them
        ai_def = self.ai_search.definitions[0]
        ai_res = self.ai_search.resources
        logging.debug(f"Created AzureAISearchTool definition: {ai_def}")
        logging.debug(f"AzureAISearchTool resources metadata: {ai_res}")

        self.tools_list.append(ai_def)
        self.tool_resources.update(ai_res)

        logging.debug(f"Final tools_list: {self.tools_list}")
        logging.debug(f"Final tool_resources: {self.tool_resources}")


    async def initiate_agent_flow(self, user_message: str):
        """
        Sends a user message and yields streamed chunks.
        Manages thread creation/reuse entirely here, storing both
        agent_id, thread_id and full messages list in self.conversation.
        """
        logging.debug(f"invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        agent_id = conv.get("agent_id")
        thread_id = conv.get("thread_id")
        logging.debug(f"Current conversation state: agent_id={agent_id}, thread_id={thread_id}")

        async with self.project_client as project_client:
            # ------------------------
            # 1) Create or reuse agent
            # ------------------------
            if agent_id:
                logging.debug("agent_id exists; calling update_agent(...)")
                agent = await project_client.agents.update_agent(
                    model=self.model_name,
                    name="gpt-rag-agent",
                    instructions=await self._read_prompt("main"),
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )
                logging.info(f"Reused agent with ID: {agent.id}")
            else:
                logging.debug("agent_id not found; calling create_agent(...)")
                agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name="gpt-rag-agent",
                    instructions=await self._read_prompt("main"),
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )
                logging.info(f"Created new agent with ID: {agent.id}")

            conv["agent_id"] = agent.id
            logging.debug(f"Stored conv['agent_id'] = {agent.id}")

            # --------------------------
            # 2) Create or reuse thread
            # --------------------------
            if thread_id:
                logging.debug(f"thread_id exists; calling get(thread_id={thread_id})")
                thread = await project_client.agents.threads.get(thread_id)
                logging.info(f"Reused thread with ID: {thread.id}")
            else:
                logging.debug("thread_id not found; calling create()")
                thread = await project_client.agents.threads.create()
                logging.info(f"Created new thread with ID: {thread.id}")

            conv["thread_id"] = thread.id
            logging.debug(f"Stored conv['thread_id'] = {thread.id}")

            # ---------------------------------
            # 3) Send user message into thread
            # ---------------------------------
            logging.debug(f"Sending user message into thread {thread.id}: {user_message!r}")
            await project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            logging.debug("User message sent.")

            # -------------------------------
            # 4) Stream back the agent answer
            # -------------------------------
            logging.debug(f"About to call project_client.agents.runs.stream(...) "
                          f"for agent_id={agent.id}, thread_id={thread.id}")
            async with await project_client.agents.runs.stream(
                thread_id=thread.id,
                agent_id=agent.id,
                event_handler=self.event_handler
            ) as stream:
                logging.debug("Entered streaming context; beginning to iterate over events...")
                async for event_type, event_data, raw in stream:
                    # Log *every* event_type for debugging
                    logging.debug(f"Stream event: type={event_type}, event_data={event_data}, raw={raw}")

                    if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                        chunk = raw or "".join(event_data.text)
                        # logging.debug(f"Yielding partial chunk: {chunk!r}")
                        yield chunk

                    elif event_type == "thread.run.failed":
                        err = event_data.last_error.message
                        logging.error(f"Stream encountered failure: {err}")
                        raise Exception(err)

                logging.debug("Streaming context closed (the run is complete).")

            # --------------------------------------------------
            # 5) After streaming, list *all* messages in the thread
            # --------------------------------------------------
            logging.debug("Fetching all messages from thread in ascending order...")
            conv["messages"] = []
            messages = project_client.agents.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING
            )
            async for msg in messages:
                if isinstance(msg.content[-1], MessageTextContent):
                    text_val = msg.content[-1].text.value
                    logging.debug(f"Retrieved message in thread: role={msg.role}, text={text_val!r}")
                    conv["messages"].append({
                        "role": msg.role,
                        "text": text_val
                    })

            logging.debug(f"Final conversation messages: {conv['messages']}")

            # --------------------------------------------------
            # 6) (OPTIONAL) Clean up the agent if desired
            # --------------------------------------------------
            # logging.debug(f"Deleting agent with ID: {agent.id}")
            # await project_client.agents.delete_agent(agent.id)
            # logging.debug("Agent deletion complete.")


class EventHandler(AsyncAgentEventHandler[str]):
    """
    Handles events emitted during the agent run lifecycle,
    converting each into a human-readable string.
    """

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        """
        Called when a partial message is received.
        :param delta: Chunk of the message text.
        :return: The text chunk.
        """
        logging.debug(f"EventHandler.on_message_delta called with delta={delta!r}")
        text = delta.text

        # Collect annotation objects, if any
        raw = getattr(delta, "delta", None)
        annotations = []
        if raw:
            for piece in getattr(raw, "content", []):
                txt = getattr(piece, "text", None)
                if not txt:
                    continue
                anns = getattr(txt, "annotations", None)
                if not anns:
                    continue
                annotations.extend(anns)

        for ann in annotations:
            if isinstance(ann, MessageDeltaTextUrlCitationAnnotation) and "url_citation" in ann:
                info = ann["url_citation"]
                placeholder = ann["text"]
            else:
                continue
            url = info.get("url")
            title = info.get("title", url)
            if url and placeholder:
                text = text.replace(placeholder, f"[{title}]({url})")

        # logging.trace(f"on_message_delta returning text={text!r}")
        return text

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        """
        Called when a new thread message object is created.
        :param message: The ThreadMessage instance.
        :return: Summary including message ID and status.
        """
        logging.debug(f"EventHandler.on_thread_message called: ID={message.id}, status={message.status}")
        return f"Thread message created: ID={message.id}, status={message.status}"

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        """
        Called when a new thread run event occurs.
        :param run: The ThreadRun instance.
        :return: Summary of the run status.
        """
        logging.debug(f"EventHandler.on_thread_run called: status={run.status}")
        return f"Thread run status: {run.status}"

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        """
        Called at each step of the run pipeline.
        :param step: The RunStep instance.
        :return: Type and status of the step.
        """
        logging.debug(f"EventHandler.on_run_step called: type={step.type}, status={step.status}")
        return f"Run step: type={step.type}, status={step.status}"

    async def on_error(self, data: str) -> Optional[str]:
        """
        Called when an error occurs during the stream.
        :param data: Error information.
        :return: Formatted error message.
        """
        logging.debug(f"EventHandler.on_error called with data={data!r}")
        return f"Error in stream: {data}"

    async def on_done(self) -> Optional[str]:
        """
        Called when the streaming completes successfully.
        :return: Completion message.
        """
        logging.debug("EventHandler.on_done called")
        return "Streaming completed"

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> Optional[str]:
        """
        Catches any events not handled by other methods.
        :param event_type: The type identifier of the event.
        :param event_data: The raw event payload.
        :return: Description of the unhandled event.
        """
        logging.debug(f"EventHandler.on_unhandled_event called: type={event_type}, data={event_data!r}")
        return f"Unhandled event: type={event_type}, data={event_data}"
