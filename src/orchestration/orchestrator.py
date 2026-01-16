import uuid
import logging

from typing import Dict, Optional
from connectors.cosmosdb import CosmosDBClient
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
    async def create(
        cls,
        conversation_id: str = None,
        user_context: Dict = {},
        request_access_token: Optional[str] = None,
    ):
        instance = cls(conversation_id=conversation_id)

        # Keep a copy for logging/troubleshooting (do not store secrets here).
        instance.user_context = user_context or {}

        # Keep the incoming API access token in memory only.
        # Do not store it in conversation documents.
        instance.request_access_token = request_access_token

        # app configuration
        cfg = get_config()

        # agentic strategy
        agentic_strategy_name = cfg.get("AGENT_STRATEGY", "single_agent_rag")
        instance.agentic_strategy = await AgentStrategyFactory.get_strategy(agentic_strategy_name)
        if not instance.agentic_strategy:
            raise EnvironmentError("AGENT_STRATEGY must be set")

        instance.agentic_strategy.user_context = user_context

        # Provide the incoming API token to strategies that can use it for OBO.
        # This is intentionally separate from user_context to avoid persisting tokens.
        try:
            setattr(instance.agentic_strategy, "request_access_token", request_access_token)
        except Exception:
            pass

        return instance

    async def stream_response(self, ask: str, question_id: Optional[str] = None):
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

            # Info-level lifecycle log (useful even when LOG_LEVEL=INFO)
            try:
                uc = getattr(self, "user_context", {}) or {}
                principal_name = (uc.get("principal_name") or "").strip()
                principal_id = (uc.get("principal_id") or "").strip()
                principal = principal_name or principal_id or "anonymous"
                auth_mode = "authenticated" if principal != "anonymous" else "anonymous"
                logging.info(
                    "[Conversation] Started: conversation_id=%s question_id=%s auth=%s principal=%s",
                    self.conversation_id,
                    question_id or "∅",
                    auth_mode,
                    principal,
                )
            except Exception:
                # Never fail due to logging.
                pass

            # Optionally record the incoming question (id + text) for traceability
            if question_id:
                questions = conversation.get("questions") or []
                questions.append({
                    "question_id": question_id,
                    "text": ask
                })
                conversation["questions"] = questions

            # 2) Hand off the conversation dict to the strategy
            self.agentic_strategy.conversation = conversation

            # 3) Stream all chunks from the strategy
            try:
                yield f"{self.conversation_id} "
                async for chunk in self.agentic_strategy.initiate_agent_flow(ask):
                    yield chunk
            finally:
                # 4) Persist whatever the strategy has updated (e.g. thread_id)
                await self.database_client.update_document(self.database_container, self.agentic_strategy.conversation)

            logging.info(
                "[Conversation] Finished: conversation_id=%s question_id=%s",
                self.conversation_id,
                question_id or "∅",
            )

    async def save_feedback(self, feedback: Dict):
        """
        Save user feedback into the same Cosmos DB container as the conversation.
        """
        if not self.conversation_id:
            raise ValueError("Conversation ID is required to save feedback")

        # Retrieve existing conversation document
        conversation = await self.database_client.get_document(
            self.database_container,
            self.conversation_id
        )
        if conversation is None:
            raise ValueError(f"Conversation {self.conversation_id} not found in database")

        # Try to resolve question_id if the client didn't provide it
        try:
            provided_question_id = feedback.get("question_id")
            if not provided_question_id:
                questions = conversation.get("questions") or []
                resolved_question_id = None

                # 1) Prefer matching by question text (most recent first)
                fb_text = (feedback.get("text") or "").strip()
                if fb_text:
                    for q in reversed(questions):
                        if (q.get("text") or "").strip() == fb_text:
                            resolved_question_id = q.get("question_id")
                            break

                # 2) Fallback to the last question's question_id
                if not resolved_question_id and questions:
                    resolved_question_id = questions[-1].get("question_id")

                if resolved_question_id:
                    feedback["question_id"] = resolved_question_id
                else:
                    logging.warning(
                        f"Could not resolve question_id for feedback in conversation {self.conversation_id}; saving with question_id=null"
                    )
        except Exception as e:
            # Do not fail feedback saving if resolution logic errors; just log
            logging.exception("Error attempting to resolve question_id from conversation questions: %s", e)

        if "feedback" not in conversation:
            conversation["feedback"] = []
        conversation["feedback"].append(feedback)

        await self.database_client.update_document(self.database_container, conversation)
        logging.info(f"Feedback saved for conversation {self.conversation_id}")
