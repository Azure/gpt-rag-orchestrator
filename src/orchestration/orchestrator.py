import asyncio
import uuid
import logging
import time

from datetime import datetime, timezone
from typing import Dict, Optional
from connectors.cosmosdb import get_cosmosdb_client
from orchestration.conversation_compaction import (
    compact_conversation_for_persistence,
    load_conversation_compaction_config,
)
from strategies.agent_strategy_factory import AgentStrategyFactory
from strategies.base_agent_strategy import BaseAgentStrategy
from dependencies import get_config
from opentelemetry.trace import SpanKind
from telemetry import (
    AuditEmitter,
    AuditStatus,
    EventType,
    ReasonCode,
    Telemetry,
    begin_audit_request,
    end_audit_request,
)

tracer = Telemetry.get_tracer(__name__)

class Orchestrator:
    agentic_strategy = BaseAgentStrategy

    def __init__(
        self,
        conversation_id: str,
        principal_id: str = None,
        correlation_id: str | None = None,
    ):
        # initializations

        # conversation_id
        self.conversation_id = conversation_id
        self.principal_id = (principal_id or "").strip() or "anonymous"
        self.correlation_id = correlation_id

        # app configuration
        cfg = get_config()

        # database
        self.database_client = get_cosmosdb_client()
        self.database_container = cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        self.conversation_compaction_config = load_conversation_compaction_config(cfg)

    @classmethod
    async def create(
        cls,
        conversation_id: str = None,
        user_context: Dict = {},
        request_access_token: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        instance = cls(
            conversation_id=conversation_id,
            principal_id=(user_context.get("principal_id") or "").strip(),
            correlation_id=correlation_id,
        )

        # Keep a copy for logging/troubleshooting (do not store secrets here).
        instance.user_context = user_context or {}

        # Keep the incoming API access token in memory only.
        # Do not store it in conversation documents.
        instance.request_access_token = request_access_token

        # app configuration
        cfg = get_config()

        # agentic strategy
        agentic_strategy_name = cfg.get("AGENT_STRATEGY", "maf_lite")
        instance.agentic_strategy = await AgentStrategyFactory.get_strategy(agentic_strategy_name)
        if not instance.agentic_strategy:
            raise EnvironmentError("AGENT_STRATEGY must be set")

        # Best-effort: propagate incoming conversation_id (may be None here).
        if instance.agentic_strategy and hasattr(instance.agentic_strategy, "set_context"):
            instance.agentic_strategy.set_context(conversation_id)

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
            emitter = AuditEmitter.default()
            span_conversation_id = (
                emitter.pseudonymize("conversation", self.conversation_id)
                if emitter.enabled
                else self.conversation_id
            )
            if span_conversation_id:
                span.set_attribute('conversation_id', span_conversation_id)
            audit_context = None
            audit_token = None
            conversation = None
            response_capture = ""
            started_monotonic = time.monotonic()
            if emitter.enabled:
                audit_context, audit_token = begin_audit_request(self.correlation_id)
                self.correlation_id = audit_context.correlation_id
                actor_id = None
                if emitter.settings.actor_pseudonym_enabled:
                    actor_id = emitter.pseudonymize("actor", self.principal_id)
                audit_context.request_started_event_id = emitter.emit(
                    EventType.REQUEST_STARTED,
                    operation="orchestrator.stream_response",
                    status=AuditStatus.STARTED,
                    reason_code=ReasonCode.REQUEST_RECEIVED,
                    correlation_id=self.correlation_id,
                    metadata={
                        "conversation_id": emitter.pseudonymize(
                            "conversation", self.conversation_id
                        ),
                        "question_id": emitter.pseudonymize(
                            "question", question_id
                        ),
                        "actor_id": actor_id,
                        "input_count": len(ask),
                    },
                    sensitive={"prompt": ask},
                )
                strategy_name = getattr(
                    getattr(self.agentic_strategy, "strategy_type", None),
                    "value",
                    type(self.agentic_strategy).__name__,
                )
                emitter.emit(
                    EventType.ROUTE_SELECTED,
                    operation="orchestrator.select_strategy",
                    status=AuditStatus.SELECTED,
                    reason_code=ReasonCode.STRATEGY_CONFIGURED,
                    parent_event_id=audit_context.request_started_event_id,
                    metadata={
                        "decision_type": "agent_strategy",
                        "decision_value": strategy_name,
                    },
                )

            try:
                # 1) Load or create our conversation document in Cosmos
                # For anonymous users, use anonymous-{conversation_id} as partition key to avoid hot partitions
                # For authenticated users, use their principal_id
                if not self.conversation_id:
                    self.conversation_id = str(uuid.uuid4())
                    partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id
                    default_name = ask[:50] if ask else "Untitled Conversation"
                    conversation = {
                        "id": self.conversation_id,
                        "name": default_name,
                        "principal_id": partition_key,
                        "lastUpdated": datetime.now(timezone.utc).isoformat(),
                    }
                    asyncio.create_task(self.database_client.create_document(
                        self.database_container, self.conversation_id, conversation, partition_key=partition_key
                    ))
                else:
                    partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id
                    conversation = await self.database_client.get_document(
                        self.database_container, self.conversation_id, partition_key=partition_key
                    )
                    if conversation is None:
                        logging.info(f"Conversation {self.conversation_id} not found; creating new conversation")
                        default_name = ask[:50] if ask else "Untitled Conversation"
                        conversation = {
                            "id": self.conversation_id,
                            "name": default_name,
                            "principal_id": partition_key,
                            "lastUpdated": datetime.now(timezone.utc).isoformat(),
                        }
                        asyncio.create_task(self.database_client.create_document(
                            self.database_container, self.conversation_id, conversation, partition_key=partition_key
                        ))

                # Search/RAG scoping: conversation_id is finalized here when the client omitted it on create().
                # Keep strategy in sync so retrieval filters by this id (not None).
                if self.agentic_strategy and hasattr(self.agentic_strategy, "set_context"):
                    self.agentic_strategy.set_context(self.conversation_id)


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
                        "text": ask,
                        "correlation_id": self.correlation_id,
                    })
                    conversation["questions"] = questions

                # 2) Hand off the conversation dict to the strategy
                self.agentic_strategy.conversation = conversation

                # 3) Stream all chunks from the strategy
                yield f"{self.conversation_id} "
                async for chunk in self.agentic_strategy.initiate_agent_flow(ask):
                    if audit_context is not None:
                        audit_context.output_count += len(chunk)
                        audit_context.partial_output = True
                        if (
                            emitter.settings.sensitive_content_enabled
                            and "response"
                            in emitter.settings.sensitive_content_fields
                            and len(response_capture) < 2048
                        ):
                            response_capture += chunk[
                                : 2048 - len(response_capture)
                            ]
                    yield chunk

                if audit_context is not None:
                    emitter.emit(
                        EventType.OUTCOME_PRODUCED,
                        operation="orchestrator.produce_outcome",
                        status=AuditStatus.PRODUCED,
                        reason_code=ReasonCode.OUTCOME_PRODUCED,
                        parent_event_id=audit_context.request_started_event_id,
                        metadata={
                            "outcome_type": "streamed_response",
                            "output_count": audit_context.output_count,
                            "source_count": audit_context.source_events_emitted,
                            "partial_output": False,
                        },
                        sensitive={"response": response_capture},
                    )
                    emitter.emit(
                        EventType.REQUEST_COMPLETED,
                        operation="orchestrator.stream_response",
                        status=AuditStatus.COMPLETED,
                        reason_code=ReasonCode.REQUEST_COMPLETED,
                        parent_event_id=audit_context.request_started_event_id,
                        started_at=audit_context.started_at,
                        duration_ms=(time.monotonic() - started_monotonic) * 1000,
                        metadata={
                            "output_count": audit_context.output_count,
                            "source_count": audit_context.source_events_emitted,
                            "partial_output": False,
                        },
                    )

                logging.info(
                    "[Conversation] Finished: conversation_id=%s question_id=%s",
                    self.conversation_id,
                    question_id or "∅",
                )
            except (asyncio.CancelledError, GeneratorExit):
                if audit_context is not None:
                    emitter.emit(
                        EventType.REQUEST_CANCELLED,
                        operation="orchestrator.stream_response",
                        status=AuditStatus.CANCELLED,
                        reason_code=(
                            ReasonCode.PARTIAL_OUTPUT
                            if audit_context.partial_output
                            else ReasonCode.CLIENT_DISCONNECTED
                        ),
                        parent_event_id=audit_context.request_started_event_id,
                        started_at=audit_context.started_at,
                        duration_ms=(time.monotonic() - started_monotonic) * 1000,
                        metadata={
                            "output_count": audit_context.output_count,
                            "source_count": audit_context.source_events_emitted,
                            "partial_output": audit_context.partial_output,
                            "failure_type": "client_disconnect",
                        },
                    )
                raise
            except Exception:
                if audit_context is not None:
                    emitter.emit(
                        EventType.OUTCOME_REJECTED,
                        operation="orchestrator.produce_outcome",
                        status=AuditStatus.REJECTED,
                        reason_code=ReasonCode.OUTCOME_REJECTED,
                        parent_event_id=audit_context.request_started_event_id,
                        metadata={
                            "outcome_type": "streamed_response",
                            "output_count": audit_context.output_count,
                            "partial_output": audit_context.partial_output,
                            "failure_type": "exception",
                        },
                    )
                    emitter.emit(
                        EventType.REQUEST_FAILED,
                        operation="orchestrator.stream_response",
                        status=AuditStatus.FAILED,
                        reason_code=ReasonCode.REQUEST_FAILED,
                        parent_event_id=audit_context.request_started_event_id,
                        started_at=audit_context.started_at,
                        duration_ms=(time.monotonic() - started_monotonic) * 1000,
                        metadata={
                            "output_count": audit_context.output_count,
                            "source_count": audit_context.source_events_emitted,
                            "partial_output": audit_context.partial_output,
                            "failure_type": "exception",
                        },
                    )
                raise
            finally:
                if isinstance(conversation, dict):
                    # 4) Persist whatever the strategy has updated (e.g. thread_id)
                    async def persist_conversation():
                        start_time = time.time()
                        try:
                            conversation_to_persist = self._prepare_conversation_for_persistence(
                                self.agentic_strategy.conversation
                            )
                            self.agentic_strategy.conversation = conversation_to_persist
                            await self.database_client.update_document(self.database_container, conversation_to_persist)
                            logging.info(f"[Orchestrator][Timing] conversation_persist_async_done: {time.time() - start_time:.2f}s")
                        except Exception as e:
                            logging.error(f"[Orchestrator] Error asynchronously persisting conversation: {e}")

                    asyncio.create_task(persist_conversation())
                if audit_token is not None:
                    end_audit_request(audit_token)

    async def save_feedback(self, feedback: Dict):
        """
        Save user feedback into the same Cosmos DB container as the conversation.
        """
        if not self.conversation_id:
            raise ValueError("Conversation ID is required to save feedback")

        # Retrieve existing conversation document
        # For anonymous users, use anonymous-{conversation_id} as partition key
        partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id
        conversation = await self.database_client.get_document(
            self.database_container,
            self.conversation_id,
            partition_key=partition_key,
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

        conversation_to_persist = self._prepare_conversation_for_persistence(conversation)
        await self.database_client.update_document(self.database_container, conversation_to_persist)
        logging.info(f"Feedback saved for conversation {self.conversation_id}")

    def _prepare_conversation_for_persistence(self, conversation: Dict) -> Dict:
        compacted, stats = compact_conversation_for_persistence(
            conversation,
            self.conversation_compaction_config,
        )
        if stats.get("compacted"):
            logging.info(
                "[Orchestrator] conversation compacted before persistence "
                "(conversation_id=%s pruned_messages=%s pruned_questions=%s bytes=%s->%s)",
                compacted.get("id") or self.conversation_id,
                stats.get("pruned_messages"),
                stats.get("pruned_questions"),
                stats.get("original_bytes"),
                stats.get("final_bytes"),
            )
        return compacted
