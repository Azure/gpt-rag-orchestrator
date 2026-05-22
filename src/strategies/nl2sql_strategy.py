import asyncio
import logging
import re
import time
import uuid
from typing import AsyncIterator, Tuple

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
# (same pattern as single_agent_rag_strategy_v2.py — reduces noisy HTTP telemetry in agent flows)
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure",
]:
    _logger = logging.getLogger(_azure_logger)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False
    _logger.disabled = True
    _logger.handlers.clear()

from azure.ai.projects.models import PromptAgentDefinition
from semantic_kernel.agents import (
    AzureAIAgent,
    AgentGroupChat
)
from semantic_kernel.agents.strategies import TerminationStrategy

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from plugins.nl2sql.plugin import NL2SQLPlugin


class ApprovalTerminationStrategy(TerminationStrategy):
    """Terminate as soon as the assistant emits TERMINATE."""
    def __init__(self, terminator_re: re.Pattern, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._terminator_re = terminator_re

    async def should_agent_terminate(self, agent, history):
        last = history[-1].content
        return bool(self._terminator_re.search(last))


class NL2SQLStrategy(BaseAgentStrategy):
    """
    An optimized NL2SQL Retrieval-Augmented Generation strategy
    """
    def __init__(self):
        super().__init__()
        self.strategy_type = AgentStrategies.NL2SQL

        # single plugin instance
        self._nl2sql_plugin = NL2SQLPlugin()

        # precompile the terminator-cleanup regex
        self._terminator_re = re.compile(r'\bterminate\b', re.IGNORECASE)

        # placeholders for prompts (lazy-loaded)
        self._triage_prompt    = None
        self._sqlquery_prompt  = None

    async def _load_prompts(self):
        """Load and cache the three prompt templates once per instance."""
        if self._triage_prompt is None:
            self._triage_prompt      = await self._read_prompt("triage_agent")
            self._sqlquery_prompt    = await self._read_prompt("sqlquery_agent")
            self._syntetizer_prompt  = await self._read_prompt("syntetizer_agent")

    async def _create_prompt_agent_version(self, client, *, agent_name: str, instructions: str):
        return await client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=self.model_name,
                instructions=instructions,
            ),
        )

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        # ensure prompts are loaded
        await self._load_prompts()

        # open a single client/session for creation + streaming
        async with self.credential as creds, \
                   AzureAIAgent.create_client(
                       credential=creds,
                       endpoint=self.project_endpoint
                   ) as client:

            # 1) create all three agents in parallel
            suffix = uuid.uuid4().hex[:8]
            t0 = time.time()
            triage_def, sql_def, syntetizer_def = await asyncio.gather(
                self._create_prompt_agent_version(
                    client, agent_name=f"nl2sql-triage-{suffix}", instructions=self._triage_prompt
                ),
                self._create_prompt_agent_version(
                    client, agent_name=f"nl2sql-sqlquery-{suffix}", instructions=self._sqlquery_prompt
                ),
                self._create_prompt_agent_version(
                    client, agent_name=f"nl2sql-syntetizer-{suffix}", instructions=self._syntetizer_prompt
                ),
            )
            logging.info(
                f"[NL2SQL][Telemetry] Agent versions parallel setup took: {time.time() - t0:.2f}s"
            )
            agent_versions: list[Tuple[str, str, str]] = [
                (triage_def.name, triage_def.version, "triage_agent"),
                (sql_def.name, sql_def.version, "sqlquery_agent"),
                (syntetizer_def.name, syntetizer_def.version, "syntetizer_agent"),
            ]

            # 2) wrap them in AzureAIAgent objects (using keyword args!)
            triage_agent = AzureAIAgent(
                client=client,
                definition=triage_def,
                plugins=[self._nl2sql_plugin]
            )
            sqlquery_agent = AzureAIAgent(
                client=client,
                definition=sql_def,
                plugins=[self._nl2sql_plugin]
            )
            syntetizer_agent = AzureAIAgent(
                client=client,
                definition=syntetizer_def,
                plugins=[self._nl2sql_plugin]
            )

            # 3) assemble group chat with our custom terminator
            chat = AgentGroupChat(
                agents=[triage_agent, sqlquery_agent, syntetizer_agent],
                termination_strategy=ApprovalTerminationStrategy(
                    terminator_re=self._terminator_re,
                    agents=[syntetizer_agent],
                    maximum_iterations=10
                ),
            )

            try:
                # start the conversation
                await chat.add_chat_message(message=user_message)

                buffer = ""
                async for content in chat.invoke_stream():

                    if content.name == "SyntetizerAgent":

                        buffer += content.content

                        # only process once the regex matches
                        if not self._terminator_re.search(buffer):
                            continue

                        # strip the terminator and yield
                        cleaned = self._terminator_re.sub("", buffer)
                        buffer = ""
                        yield cleaned

            finally:
                # clear conversation state
                try:
                    await chat.reset()
                except Exception as e:
                    logging.warning(f"Chat reset failed: {e!r}")

                # schedule background deletions
                for agent_name, agent_version, label in agent_versions:
                    if agent_name and agent_version:
                        logging.info(
                            f"Scheduling deletion for {label} "
                            f"(name={agent_name}, version={agent_version})"
                        )
                        self._schedule_agent_deletion(agent_name, agent_version)
                    else:
                        logging.warning(f"{label} has no name/version; skipping deletion.")

    def _schedule_agent_deletion(self, agent_name: str, agent_version: str):
        """
        Fire-and-forget deletion that opens its own client/session,
        preventing “Session is closed” errors.
        """
        async def _delete():
            try:
                async with self.credential as creds, \
                           AzureAIAgent.create_client(
                               credential=creds,
                               endpoint=self.project_endpoint
                           ) as delete_client:
                    await delete_client.agents.delete_version(
                        agent_name=agent_name,
                        agent_version=agent_version,
                    )
                    logging.info(
                        f"Background deleted agent {agent_name} version {agent_version}"
                    )
            except Exception as e:
                logging.error(
                    f"Failed background deletion of {agent_name}:{agent_version}: {e!r}"
                )

        task = asyncio.create_task(_delete())
        task.add_done_callback(lambda t: t.exception())
