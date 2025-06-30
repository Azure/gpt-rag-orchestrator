import uuid
import logging

from typing import Dict
from connectors.cosmosdb import CosmosDBClient
from connectors.appconfig import AppConfigClient
from strategies.agent_strategy_factory import AgentStrategyFactory
from strategies.base_agent_strategy import BaseAgentStrategy
from dependencies import get_config
from opentelemetry.trace import SpanKind
from telemetry import Telemetry

tracer = Telemetry.get_tracer(__name__)

class Orchestrator:
    agentic_strategy = BaseAgentStrategy

    def __init__(self, conversation_id: str):
        # initializations

        # conversation_id
        self.conversation_id = conversation_id

        # app configuration
        cfg = get_config()
        
        # database
        self.database_client = CosmosDBClient()
        self.database_container = cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        
    @classmethod
    async def create(cls, conversation_id: str = None, user_context: Dict = {}):
        instance = cls(conversation_id=conversation_id)

        # app configuration
        cfg = get_config()

        # agentic strategy
        agentic_strategy_name = cfg.get("AGENT_STRATEGY", "single_agent_rag")
        instance.agentic_strategy = await AgentStrategyFactory.get_strategy(agentic_strategy_name)
        if not instance.agentic_strategy:
            raise EnvironmentError("AGENT_STRATEGY must be set")

        instance.agentic_strategy.user_context = user_context

        return instance

    async def stream_response(self, ask: str):
        logging.debug(f"Starting conversation {self.conversation_id}")

        with tracer.start_as_current_span('stream_response', kind=SpanKind.SERVER) as span:

            span.set_attribute('conversation_id', self.conversation_id)

            # 1) Load or create our conversation document in Cosmos
            if not self.conversation_id:
                self.conversation_id = str(uuid.uuid4())
                conversation = {"id": self.conversation_id}
                await self.database_client.create_document(self.database_container, self.conversation_id, conversation)
            else:
                conversation = await self.database_client.get_document(self.database_container, self.conversation_id)
                if conversation is None:
                    raise ValueError(f"Conversation {self.conversation_id} not found")

            # 2) Hand off the conversation dict to the strategy
            self.agentic_strategy.conversation = conversation

            # 3) Stream all chunks from the strategy
            try:
                async for chunk in self.agentic_strategy.initiate_agent_flow(ask):
                    yield chunk
            finally:
                # 4) Persist whatever the strategy has updated (e.g. thread_id)
                await self.database_client.update_document(self.database_container, self.agentic_strategy.conversation)

            logging.debug(f"Finished conversation {self.conversation_id}")
