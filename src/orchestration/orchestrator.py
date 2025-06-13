import uuid
import logging
import os

from connectors.cosmosdb import CosmosDBClient
from connectors.appconfig import AppConfigClient
from strategies.agent_strategy_factory import AgentStrategyFactory


class Orchestrator:
    def __init__(self, conversation_id: str):
        # initializations

        # conversation_id
        self.conversation_id = conversation_id

        # app configuration
        self.cfg = AppConfigClient()
        
        # database
        self.database_client = CosmosDBClient()
        self.database_container = self.cfg("conversationsContainer", "conversations")


        # agentic strategy
        agentic_strategy_name = self.cfg.get("agentStrategy", "single_agent_rag")
        self.agentic_strategy = AgentStrategyFactory.get_strategy(agentic_strategy_name)
        if not self.agentic_strategy or not self.database_container:
            raise EnvironmentError("AGENTIC_STRATEGY and COSMOS_DB_CONTAINER must be set")        

    async def stream_response(self, ask: str):
        logging.debug(f"Starting conversation {self.conversation_id}")

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
