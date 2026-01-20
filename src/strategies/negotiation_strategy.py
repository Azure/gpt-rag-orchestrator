"""
Negotiation Simulation Strategy using Azure AI Foundry Agent Service.

This strategy implements a negotiation practice system with:
- Router agent for intent detection (start/continue/end simulation)
- Multiple buyer personas (CFO, Director, Manager, Analyst)
- Feedback coach for post-simulation analysis
- Scenario templates for different negotiation contexts
"""

import json
import logging
import time
from typing import Optional, Dict, List, Any

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    logger = logging.getLogger(_azure_logger)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True
    logger.handlers.clear()

from azure.ai.agents.models import ListSortOrder, MessageTextContent
from azure.ai.projects.aio import AIProjectClient

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from dependencies import get_config


class NegotiationStrategy(BaseAgentStrategy):
    """
    Negotiation simulation with router + persona agents.

    This strategy enables users to practice negotiation skills with different
    buyer personas (CFO, Director, Manager, Analyst) and receive feedback.
    """

    PERSONAS = {
        "cfo": "C-Suite Executive (CFO)",
        "director": "Senior Director",
        "manager": "Manager",
        "analyst": "Procurement Analyst",
        "feedback_coach": "Feedback Coach"
    }

    SCENARIOS = {
        "saas_software": "Enterprise SaaS Software Sale",
        "consulting": "Consulting Engagement",
        "hardware": "Hardware/Equipment Procurement",
        "services": "Professional Services Contract"
    }

    def __init__(self):
        """Initialize the negotiation strategy."""
        super().__init__()

        logging.debug("[Init] Initializing NegotiationStrategy...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.NEGOTIATION

        # No tools for this version
        self.tools_list = []
        self.tool_resources = {}

        logging.debug("[Init] NegotiationStrategy initialized")

    def _create_project_client(self) -> AIProjectClient:
        """
        Create a fresh AIProjectClient instance.

        The base class stores a single client instance, but using it with
        `async with` multiple times fails after the first use closes the
        HTTP transport. This method creates a new client for each usage.
        """
        return AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential
        )

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiate the negotiation agent flow.

        Steps:
        1. Detect intent using router
        2. Handle based on intent (start/continue/end simulation or general)
        3. Stream response
        4. Update conversation state
        """
        flow_start = time.time()
        logging.debug(f"[Agent Flow] initiate_agent_flow called with user_message: {user_message!r}")

        conv = self.conversation
        current_state = conv.get("simulation_state", "inactive")

        # Step 1: Detect intent
        intent = await self._detect_intent(user_message, current_state)
        logging.info(f"[Agent Flow] Detected intent: {intent}")

        # Step 2: Handle based on intent
        if intent["intent"] == "start_simulation":
            persona = intent.get("persona", "manager")
            scenario = intent.get("scenario")

            # Initialize simulation state
            conv["simulation_state"] = "active"
            conv["current_persona"] = persona
            conv["current_scenario"] = scenario
            conv["simulation_history"] = []

            logging.info(f"[Agent Flow] Starting simulation with persona: {persona}, scenario: {scenario}")

            async for chunk in self._run_persona(persona, user_message, is_start=True, scenario=scenario):
                yield chunk

        elif intent["intent"] == "end_simulation":
            conv["simulation_state"] = "feedback"

            logging.info("[Agent Flow] Ending simulation, generating feedback")

            async for chunk in self._run_feedback_coach():
                yield chunk

            # Reset simulation state
            conv["simulation_state"] = "inactive"
            conv["current_persona"] = None
            conv["current_scenario"] = None

            # Cleanup persona agent if exists
            await self._cleanup_simulation()

        elif intent["intent"] == "continue_simulation":
            persona = conv.get("current_persona", "manager")
            scenario = conv.get("current_scenario")

            logging.info(f"[Agent Flow] Continuing simulation with persona: {persona}")

            async for chunk in self._run_persona(persona, user_message, is_start=False, scenario=scenario):
                yield chunk

        else:  # general
            logging.info("[Agent Flow] Handling as general conversation")

            async for chunk in self._run_general_assistant(user_message):
                yield chunk

        logging.info(f"[Agent Flow] Total flow time: {round(time.time() - flow_start, 2)}s")

    async def _detect_intent(self, user_message: str, current_state: str) -> Dict[str, Any]:
        """
        Detect user intent using the router agent.

        Returns:
            dict: {
                "intent": "start_simulation" | "continue_simulation" | "end_simulation" | "general",
                "persona": "cfo" | "director" | "manager" | "analyst" | None,
                "scenario": "saas_software" | "consulting" | "hardware" | "services" | None,
                "reason": "..."
            }
        """
        try:
            async with self._create_project_client() as project_client:
                # Read router prompt
                router_instructions = await self._read_prompt(
                    "router",
                    placeholders={"current_state": current_state}
                )

                # Create router agent
                router_agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name="negotiation-router",
                    instructions=router_instructions,
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )

                try:
                    # Create thread and send message
                    thread = await project_client.agents.threads.create()

                    await project_client.agents.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=user_message
                    )

                    # Run and collect response
                    full_response = ""
                    async with await project_client.agents.runs.stream(
                        thread_id=thread.id,
                        agent_id=router_agent.id
                    ) as stream:
                        async for event_type, event_data, raw in stream:
                            if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                                chunk = event_data.text
                                if chunk:
                                    full_response += chunk

                    # Parse JSON response
                    intent_data = self._parse_intent_response(full_response, current_state)

                    # Cleanup
                    await project_client.agents.threads.delete(thread.id)

                    return intent_data

                finally:
                    await project_client.agents.delete_agent(router_agent.id)

        except Exception as e:
            logging.error(f"[Router] Intent detection failed: {e}", exc_info=True)
            # Fallback based on current state
            if current_state == "active":
                return {"intent": "continue_simulation", "persona": None, "scenario": None, "reason": "fallback"}
            return {"intent": "general", "persona": None, "scenario": None, "reason": "fallback"}

    def _parse_intent_response(self, response: str, current_state: str) -> Dict[str, Any]:
        """Parse the router's JSON response into intent data."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle potential markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            data = json.loads(response)

            return {
                "intent": data.get("intent", "general"),
                "persona": data.get("persona"),
                "scenario": data.get("scenario"),
                "reason": data.get("reason", "")
            }
        except json.JSONDecodeError:
            logging.warning(f"[Router] Failed to parse JSON response: {response}")
            # Fallback based on current state
            if current_state == "active":
                return {"intent": "continue_simulation", "persona": None, "scenario": None, "reason": "parse_fallback"}
            return {"intent": "general", "persona": None, "scenario": None, "reason": "parse_fallback"}

    async def _run_persona(
        self,
        persona: str,
        user_message: str,
        is_start: bool = False,
        scenario: Optional[str] = None
    ):
        """
        Run a persona agent and stream response.

        Args:
            persona: The persona to use (cfo, director, manager, analyst)
            user_message: The user's message
            is_start: Whether this is the start of a simulation
            scenario: Optional scenario context
        """
        conv = self.conversation

        async with self._create_project_client() as project_client:
            if is_start:
                # Build persona instructions with scenario context
                instructions = await self._build_persona_instructions(persona, scenario)

                # Create new agent for this simulation
                agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name=f"negotiation-{persona}",
                    instructions=instructions,
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )
                conv["persona_agent_id"] = agent.id

                # Create new thread
                thread = await project_client.agents.threads.create()
                conv["persona_thread_id"] = thread.id

                logging.info(f"[Persona] Created agent {agent.id} and thread {thread.id}")
            else:
                # Reuse existing agent and thread
                agent_id = conv.get("persona_agent_id")
                thread_id = conv.get("persona_thread_id")

                if not agent_id or not thread_id:
                    logging.error("[Persona] No existing agent/thread found")
                    raise Exception("Simulation state corrupted - no agent/thread found")

                agent = await project_client.agents.get_agent(agent_id)
                thread = await project_client.agents.threads.get(thread_id)

            # Send user message
            await project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            # Stream response
            full_response = ""
            async with await project_client.agents.runs.stream(
                thread_id=thread.id,
                agent_id=agent.id
            ) as stream:
                async for event_type, event_data, raw in stream:
                    if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                        chunk = event_data.text
                        if chunk:
                            full_response += chunk
                            yield chunk

                    if event_type == "thread.run.failed":
                        err = event_data.last_error.message
                        logging.error(f"[Persona] Run failed: {err}")
                        raise Exception(err)

            # Record in simulation history
            conv["simulation_history"].append({
                "role": "user",
                "text": user_message
            })
            conv["simulation_history"].append({
                "role": "assistant",
                "persona": persona,
                "text": full_response
            })

            logging.info(f"[Persona] Completed response, history has {len(conv['simulation_history'])} entries")

    async def _build_persona_instructions(self, persona: str, scenario: Optional[str] = None) -> str:
        """Build persona instructions with optional scenario context."""
        # Read persona prompt
        instructions = await self._read_prompt(persona)

        # Add scenario context if provided
        if scenario:
            try:
                scenario_context = await self._read_prompt(f"scenarios/{scenario}")
                instructions = f"{instructions}\n\n## Scenario Context\n{scenario_context}"
            except FileNotFoundError:
                logging.warning(f"[Persona] Scenario '{scenario}' not found, using persona only")

        return instructions

    async def _run_feedback_coach(self):
        """Run the feedback coach to analyze the simulation."""
        conv = self.conversation
        simulation_history = conv.get("simulation_history", [])

        if not simulation_history:
            yield "No simulation history found to analyze. Please complete a negotiation simulation first."
            return

        # Format history for feedback
        history_text = self._format_simulation_history(simulation_history)
        persona = conv.get("current_persona", "unknown")
        scenario = conv.get("current_scenario", "general negotiation")

        async with self._create_project_client() as project_client:
            # Read feedback coach prompt
            instructions = await self._read_prompt(
                "feedback_coach",
                placeholders={
                    "simulation_history": history_text,
                    "persona": self.PERSONAS.get(persona, persona),
                    "scenario": self.SCENARIOS.get(scenario, scenario) if scenario else "general negotiation"
                }
            )

            # Create feedback agent
            feedback_agent = await project_client.agents.create_agent(
                model=self.model_name,
                name="negotiation-feedback-coach",
                instructions=instructions,
                tools=self.tools_list,
                tool_resources=self.tool_resources
            )

            try:
                thread = await project_client.agents.threads.create()

                await project_client.agents.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content="Please provide feedback on my negotiation simulation."
                )

                async with await project_client.agents.runs.stream(
                    thread_id=thread.id,
                    agent_id=feedback_agent.id
                ) as stream:
                    async for event_type, event_data, raw in stream:
                        if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                            chunk = event_data.text
                            if chunk:
                                yield chunk

                        if event_type == "thread.run.failed":
                            err = event_data.last_error.message
                            logging.error(f"[Feedback] Run failed: {err}")
                            raise Exception(err)

                await project_client.agents.threads.delete(thread.id)

            finally:
                await project_client.agents.delete_agent(feedback_agent.id)

    def _format_simulation_history(self, history: List[Dict]) -> str:
        """Format simulation history for feedback analysis."""
        formatted = []
        for entry in history:
            role = entry.get("role", "unknown")
            text = entry.get("text", "")
            persona = entry.get("persona", "")

            if role == "user":
                formatted.append(f"**User (Salesperson):** {text}")
            else:
                persona_name = self.PERSONAS.get(persona, persona)
                formatted.append(f"**{persona_name}:** {text}")

        return "\n\n".join(formatted)

    async def _run_general_assistant(self, user_message: str):
        """Handle general conversation outside of simulation."""
        conv = self.conversation
        thread_id = conv.get("thread_id")

        async with self._create_project_client() as project_client:
            # Get or create thread
            if thread_id:
                try:
                    thread = await project_client.agents.threads.get(thread_id)
                except Exception:
                    thread = await project_client.agents.threads.create()
                    conv["thread_id"] = thread.id
            else:
                thread = await project_client.agents.threads.create()
                conv["thread_id"] = thread.id

            # Read general assistant prompt
            instructions = await self._read_prompt("general")

            # Create general assistant agent
            agent = await project_client.agents.create_agent(
                model=self.model_name,
                name="negotiation-assistant",
                instructions=instructions,
                tools=self.tools_list,
                tool_resources=self.tool_resources
            )

            try:
                await project_client.agents.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=user_message
                )

                async with await project_client.agents.runs.stream(
                    thread_id=thread.id,
                    agent_id=agent.id
                ) as stream:
                    async for event_type, event_data, raw in stream:
                        if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                            chunk = event_data.text
                            if chunk:
                                yield chunk

                        if event_type == "thread.run.failed":
                            err = event_data.last_error.message
                            logging.error(f"[General] Run failed: {err}")
                            raise Exception(err)

                # Consolidate conversation history
                await self._consolidate_conversation_history(project_client, thread.id)

            finally:
                await project_client.agents.delete_agent(agent.id)

    async def _consolidate_conversation_history(self, project_client, thread_id: str):
        """Fetch and consolidate conversation history from thread."""
        try:
            logging.debug("[Agent Flow] Consolidating conversation history")
            conv = self.conversation
            conv["messages"] = []

            messages = project_client.agents.messages.list(
                thread_id=thread_id,
                order=ListSortOrder.ASCENDING
            )

            msg_count = 0
            async for msg in messages:
                if not msg.content:
                    continue

                last_content = msg.content[-1]
                if isinstance(last_content, MessageTextContent):
                    text_val = last_content.text.value
                    msg_count += 1
                    conv["messages"].append({
                        "role": msg.role,
                        "text": text_val
                    })

            logging.info(f"[Agent Flow] Retrieved {msg_count} messages")

            if self.user_context:
                conv['user_context'] = self.user_context
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to consolidate history: {e}", exc_info=True)

    async def _cleanup_simulation(self):
        """Delete persona agent when simulation ends."""
        conv = self.conversation
        agent_id = conv.get("persona_agent_id")
        thread_id = conv.get("persona_thread_id")

        if agent_id or thread_id:
            try:
                async with self._create_project_client() as project_client:
                    if thread_id:
                        try:
                            await project_client.agents.threads.delete(thread_id)
                            logging.debug(f"[Cleanup] Deleted thread: {thread_id}")
                        except Exception as e:
                            logging.warning(f"[Cleanup] Failed to delete thread: {e}")

                    if agent_id:
                        try:
                            await project_client.agents.delete_agent(agent_id)
                            logging.debug(f"[Cleanup] Deleted agent: {agent_id}")
                        except Exception as e:
                            logging.warning(f"[Cleanup] Failed to delete agent: {e}")
            except Exception as e:
                logging.error(f"[Cleanup] Cleanup failed: {e}", exc_info=True)

        # Clear state
        conv["persona_agent_id"] = None
        conv["persona_thread_id"] = None
        conv["simulation_history"] = []
