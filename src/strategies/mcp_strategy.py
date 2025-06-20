import logging
from typing import Any, Optional

from azure.identity import get_bearer_token_provider

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

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

from connectors.appconfig import AppConfigClient
from dependencies import get_config

# -----------------------------------------------------------------------------
# Be sure to configure the root logger at DEBUG level somewhere early in your app,
# e.g. in your main entrypoint:
#
#    logging.basicConfig(level=logging.DEBUG, 
#                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
#
# That way, all of the logging.debug(...) calls below will actually show up.
# -----------------------------------------------------------------------------

class McpStrategy(BaseAgentStrategy):
    """
    Implements a MCP Server strategy. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()
 
        # Force all logs at DEBUG or above to appear
        logging.debug("Initializing McpStrategy...")

        # Event handler for streaming responses
        self.strategy_type = AgentStrategies.MCP
        self.event_handler = EventHandler()

        cfg = get_config()

        # Agent Tools Initialization Section
        # =========================================================
        
        # Allow the user to specify an existing agent ID
        self.existing_agent_id = cfg.get("AGENT_ID") or None

        # Agent Tools Initialization Section
        # =========================================================
        self.tools_list = []
        self.tool_resources = {}

        # Add an MCP Server tool if configured
        mcp_server_url = cfg.get("MCP_APP_ENDPOINT", default="http://localhost:5000")
        mcp_server_timeout = cfg.get("MCP_CLIENT_TIMEOUT", default=600)
        mcp_server_api_key = cfg.get("MCP_SERVER_APIKEY", default=None)
        mcp_server_transport = cfg.get("MCP_SERVER_TRANSPORT", default="sse")

        if mcp_server_url:
            logging.debug(f"Adding MCP Server tool with URL: {mcp_server_url}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            if mcp_server_api_key:
                logging.debug("MCP Server API key provided; adding to headers.")
                headers["X-API-KEY"] = f"{mcp_server_api_key}"

            if mcp_server_transport == "sse":
                # Use the MCPSsePlugin for SSE transport
                logging.debug("Using MCPSsePlugin for MCP Server with SSE transport.")
                plugin = MCPSsePlugin(
                    name="McpServerPlugin",
                    url=mcp_server_url,
                    headers=headers,
                    timeout=mcp_server_timeout
                )

                self.tools_list.append(plugin)
                
        else:
            logging.debug("No MCP Server URL provided; skipping MCP Server tool initialization.")
        
        logging.debug(f"Final tools_list: {self.tools_list}")
        logging.debug(f"Final tool_resources: {self.tool_resources}")

        self.kernel = Kernel()

        token_provider = get_bearer_token_provider(
            cfg.credential,
            "https://cognitiveservices.azure.com/.default"
        )
 
        self.kernel.add_service(AzureChatCompletion(
            service_id="chat",
            deployment_name=cfg.get("CHAT_DEPLOYMENT_NAME"),
            endpoint=cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT"),
            ad_token_provider= token_provider
        ))

        self.agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="MultiPluginAgent",
            plugins=self.tools_list
        )


    async def initiate_agent_flow(self, user_message: str):
        """
        Sends a user message and yields streamed chunks.
        Manages thread creation/reuse entirely here, storing both
        agent_id, thread_id and full messages list in self.conversation.
        """
        logging.debug(f"invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")
        logging.debug(f"Current conversation state: thread_id={thread_id}")

        conv["agent_id"] = self.agent.id
        
        response = await self.agent.get_response(messages=user_message)

        yield response
       
        conv["messages"] = []
        conv["messages"].append({
            "role": "system",
            "text": response.message.content
        })
        
        logging.debug(f"Final conversation messages: {conv['messages']}")

        

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
