import logging
from typing import AsyncIterator, Optional, Dict, List

import asyncio
from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    MessageDeltaChunk,
    ThreadRun,
    SubmitToolApprovalAction,
    RequiredMcpToolCall,
    ToolApproval,
    McpTool,
    ListSortOrder,
    RunStep,
)
from azure.ai.agents.models import ToolSet  # If not present in your version, remove and revert to tools=

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from dependencies import get_config


class _SimpleEventHandler(AsyncAgentEventHandler[str]):
    """
    Captures deltas and run status; tool approval handled in strategy.
    """
    def __init__(self):
        super().__init__()
        self.last_run_status: Optional[str] = None
        self.last_run_id: Optional[str] = None
        self.run_failed_error: Optional[str] = None

    async def on_message_delta(self, delta: MessageDeltaChunk):
        if hasattr(delta, "text") and delta.text:
            return "".join(delta.text)
        return None

    async def on_thread_run(self, run: ThreadRun):
        self.last_run_status = getattr(run, "status", None)
        self.last_run_id = getattr(run, "id", None)
        if self.last_run_status == "failed":
            self.run_failed_error = getattr(getattr(run, "last_error", None), "message", "Unknown run failure")
        return None

    async def on_run_step(self, step: RunStep):
        return None


class McpStrategy(BaseAgentStrategy):
    """
    MCP strategy (async) using non-stream tool approval:
    - Stream initial run
    - On requires_action -> approve tools -> submit outputs (non-stream)
    - Poll run to completion
    - Fallback fetch final assistant message if no deltas
    """

    def __init__(self):
        super().__init__()
        cfg = get_config()
        self.strategy_type = AgentStrategies.MCP

        self.mcp_server_url = cfg.get("MCP_SERVER_URL",  "https://gitmcp.io/Azure/azure-rest-api-specs") #"https://gitmcp.io/Azure/azure-rest-api-specs", "https://mcp-demo.kindsky-e7ac46f3.westus3.azurecontainerapps.io",  "http://127.0.0.1:8000/mcp"
        self.mcp_server_label = cfg.get("MCP_SERVER_LABEL", "mcp_server")
        self.mcp_allowed_tools_raw = cfg.get("MCP_ALLOWED_TOOLS", "")
        self.mcp_headers_raw = cfg.get("MCP_HEADERS", "")
        self.mcp_approval_mode = cfg.get("MCP_APPROVAL_MODE", "auto").lower()
        self.existing_agent_id: Optional[str] = cfg.get("MCP_AGENT_ID", "") or None
        self.prompt_name = cfg.get("MCP_PROMPT_NAME", cfg.get("PLAIN_CHAT_PROMPT_NAME", "system"))
        self.max_poll_seconds = int(cfg.get("MCP_POLL_MAX_SECONDS", 30))
        self.poll_interval = float(cfg.get("MCP_POLL_INTERVAL_SECONDS", 1.0))

        # MCP tool + toolset
        self.mcp_tool = McpTool(
            server_label=self.mcp_server_label,
            server_url=self.mcp_server_url,
            allowed_tools=[]
        )
        for t in [x.strip() for x in self.mcp_allowed_tools_raw.split(",") if x.strip()]:
            try:
                self.mcp_tool.allow_tool(t)
            except Exception as e:
                logging.warning(f"[MCP] Could not allow tool '{t}': {e!r}")

        for k, v in self._parse_headers(self.mcp_headers_raw).items():
            try:
                self.mcp_tool.update_headers(k, v)
            except Exception as e:
                logging.warning(f"[MCP] header {k} failed: {e!r}")

        if self.mcp_approval_mode not in ("auto", "never", "manual"):
            logging.warning(f"[MCP] Unknown approval mode '{self.mcp_approval_mode}', defaulting to 'auto'")
            self.mcp_approval_mode = "auto"

        if self.mcp_approval_mode == "never":
            try:
                self.mcp_tool.set_approval_mode("never")
            except Exception as e:
                logging.warning(f"[MCP] Could not set approval_mode=never: {e!r}")

        # ToolSet (if available)
        self.toolset = ToolSet()
        self.toolset.add(self.mcp_tool)

        self.conversation = {}

    @staticmethod
    def _parse_headers(raw: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for part in [p.strip() for p in raw.split(";") if p.strip()]:
            if "=" in part:
                k, v = part.split("=", 1)
                headers[k.strip()] = v.strip()
        return headers

    @classmethod
    async def create(cls):
        return cls()

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        logging.info(f"[MCP] initiate_agent_flow: {user_message!r}")

        conv = self.conversation or {}
        if "messages" not in conv or not isinstance(conv.get("messages"), list):
            conv["messages"] = []
        self.conversation = conv

        async with self.project_client as project_client:
            thread = await self._get_or_create_thread(project_client, conv.get("thread_id"))
            conv["thread_id"] = thread.id

            agent, created = await self._get_or_create_agent(project_client)
            conv["agent_id"] = agent.id

            logging.info(f"[MCP] Thread={thread.id} Agent={agent.id} Server={self.mcp_server_url}")

            await project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            handler = _SimpleEventHandler()
            collected: List[str] = []
            approvals_needed = False
            captured_run_id: Optional[str] = None

            # Initial streaming pass
            async with await project_client.agents.runs.stream(
                thread_id=thread.id,
                agent_id=agent.id,
                event_handler=handler,
                tool_resources=self.mcp_tool.resources
            ) as stream:
                async for event_type, event_data, raw in stream:
                    if event_type == "thread.message.delta":
                        text_piece = await handler.on_message_delta(event_data)
                        if text_piece:
                            collected.append(text_piece)
                            yield text_piece
                    elif event_type == "thread.run.requires_action":
                        approvals_needed = True
                        captured_run_id = getattr(event_data, "id", None)
                    elif event_type == "thread.run.failed":
                        err = getattr(getattr(event_data, "last_error", None), "message", "Unknown run failure")
                        raise RuntimeError(err)

            # Handle approvals after initial stream closes
            if approvals_needed and captured_run_id:
                logging.info(f"[MCP] Run {captured_run_id} requires tool approval cycle.")
                run_obj = await project_client.agents.runs.get(thread_id=thread.id, run_id=captured_run_id)
                approvals = self._build_tool_approvals(run_obj)
                if approvals:
                    await project_client.agents.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=captured_run_id,
                        tool_approvals=approvals
                    )
                    logging.info(f"[MCP] Submitted {len(approvals)} tool approvals.")
                    # Poll until completion
                    await self._poll_run_completion(project_client, thread.id, captured_run_id)
                else:
                    logging.warning("[MCP] No approvable tool calls found; cancelling run.")
                    try:
                        await project_client.agents.runs.cancel(thread_id=thread.id, run_id=captured_run_id)
                    except Exception as e:
                        logging.error(f"[MCP] Cancel failed: {e!r}")

            # Final answer fallback if we have no deltas (or only greeting)
            final_response = "".join(collected).strip()
            if not final_response:
                final_response = await self._fetch_latest_assistant(project_client, thread.id) or "[MCP] No response generated."
                yield final_response

            # Persist conversation
            conv["last_response"] = final_response
            conv["messages"].append({"role": "user", "content": user_message})
            conv["messages"].append({"role": "assistant", "content": final_response})

            if created:
                await self._safe_delete_agent(project_client, agent.id)

    async def _poll_run_completion(self, project_client, thread_id: str, run_id: str):
        """
        Poll run status until terminal state or timeout.
        """
        deadline = asyncio.get_event_loop().time() + self.max_poll_seconds
        while asyncio.get_event_loop().time() < deadline:
            run = await project_client.agents.runs.get(thread_id=thread_id, run_id=run_id)
            status = getattr(run, "status", None)
            if status in ("completed", "failed", "cancelled"):
                logging.info(f"[MCP] Run {run_id} terminal status={status}")
                if status == "failed":
                    err = getattr(getattr(run, "last_error", None), "message", "Unknown run failure")
                    logging.error(f"[MCP] Run failed after approval: {err}")
                return
            await asyncio.sleep(self.poll_interval)
        logging.warning(f"[MCP] Poll timeout ({self.max_poll_seconds}s) waiting for run {run_id} completion.")

    def _build_tool_approvals(self, run: ThreadRun) -> List[ToolApproval]:
        if not isinstance(run.required_action, SubmitToolApprovalAction):
            return []
        if self.mcp_approval_mode == "never":
            logging.info("[MCP] Approval mode 'never' - skipping approvals.")
            return []
        approvals: List[ToolApproval] = []
        for tc in (run.required_action.submit_tool_approval.tool_calls or []):
            if isinstance(tc, RequiredMcpToolCall):
                try:
                    approvals.append(
                        ToolApproval(
                            tool_call_id=tc.id,
                            approve=True,
                            headers=self.mcp_tool.headers
                        )
                    )
                    logging.info(f"[MCP] Approving tool call id={tc.id} name={tc.name}")
                except Exception as e:
                    logging.error(f"[MCP] Error building approval for {tc.id}: {e!r}")
        return approvals

    async def _fetch_latest_assistant(self, project_client, thread_id: str) -> Optional[str]:
        try:
            msgs = [m async for m in project_client.agents.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)]
            for m in msgs:
                if m.role == "assistant":
                    if getattr(m, "text_messages", None):
                        last = m.text_messages[-1]
                        return getattr(getattr(last, "text", None), "value", None)
                    if getattr(m, "content", None):
                        return m.content
            return None
        except Exception as e:
            logging.error(f"[MCP] Fetch assistant message failed: {e!r}")
            return None

    async def _get_or_create_thread(self, project_client, thread_id: Optional[str]):
        if thread_id:
            try:
                return await project_client.agents.threads.get(thread_id)
            except Exception as e:
                logging.warning(f"[MCP] Reuse thread failed {thread_id}: {e!r}")
        return await project_client.agents.threads.create()

    async def _get_or_create_agent(self, project_client):
        if self.existing_agent_id:
            try:
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                return agent, False
            except Exception as e:
                logging.warning(f"[MCP] Fetch existing agent failed {self.existing_agent_id}: {e!r}")

        instructions = await self._load_instructions()
        # If ToolSet unsupported, replace toolset= with tools=self.mcp_tool.definitions
        try:
            agent = await project_client.agents.create_agent(
                model=self.model_name,
                name="MCPAgentAsync",
                instructions=instructions,
                toolset=self.toolset
            )
        except TypeError:
            # Fallback for older SDK
            agent = await project_client.agents.create_agent(
                model=self.model_name,
                name="MCPAgentAsync",
                instructions=instructions,
                tools=self.mcp_tool.definitions
            )
        return agent, True

    async def _load_instructions(self) -> str:
        try:
            return await self._read_prompt(self.prompt_name)
        except Exception:
            return "You are an MCP-enabled agent. Use MCP tools when helpful."

    async def _safe_delete_agent(self, project_client, agent_id: str):
        try:
            await project_client.agents.delete_agent(agent_id)
        except Exception as e:
            logging.warning(f"[MCP] Delete agent failed: {e!r}")