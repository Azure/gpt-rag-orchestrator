import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Optional

from agent_framework import ChatAgent, ChatMessage

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from connectors.openai_chat_client import OpenAIChatClient
from plugins.nl2sql.plugin import NL2SQLPlugin


class NL2SQLStrategy(BaseAgentStrategy):
    """NL2SQL strategy backed by Microsoft Agent Framework and local tools."""

    _ANSWERED_MARKER = "QUESTION_ANSWERED"
    _DATASOURCE_MARKER = "DATASOURCE_SELECTED"
    _MAX_SCHEMA_TABLES = 8

    def __init__(self):
        super().__init__()
        self.strategy_type = AgentStrategies.NL2SQL
        self._nl2sql_plugin = NL2SQLPlugin()
        self._chat_client: Optional[OpenAIChatClient] = None
        self._sync_credential = self.cfg.credential
        self.max_completion_tokens = int(self.cfg.get("MAX_COMPLETION_TOKENS", 4096))
        self.reasoning_effort = self.cfg.get("REASONING_EFFORT", "medium")
        self.history_max_messages = int(self.cfg.get("CHAT_HISTORY_MAX_MESSAGES", 6))
        self._triage_prompt: Optional[str] = None
        self._sqlquery_prompt: Optional[str] = None
        self._syntetizer_prompt: Optional[str] = None

    def _get_or_create_chat_client(self) -> OpenAIChatClient:
        if self._chat_client is None:
            self._chat_client = OpenAIChatClient(
                azure_endpoint=self.account_endpoint,
                model_deployment_name=self.model_name,
                credential=self._sync_credential,
                api_version=self.openai_api_version,
            )
        return self._chat_client

    async def _load_prompts(self) -> None:
        if self._triage_prompt is None:
            self._triage_prompt = await self._read_prompt("triage_agent")
            self._sqlquery_prompt = await self._read_prompt("sqlquery_agent")
            self._syntetizer_prompt = await self._read_prompt("syntetizer_agent")

    async def _run_agent(self, instructions: str, message: str, *, max_tokens: int = 1200) -> str:
        chunks = []
        async for chunk in self._stream_agent(instructions, message, max_tokens=max_tokens):
            chunks.append(chunk)
        return "".join(chunks).strip()

    async def _stream_agent(
        self,
        instructions: str,
        message: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        chat_client = self._get_or_create_chat_client()
        async with ChatAgent(chat_client=chat_client, instructions=instructions) as agent:
            thread = agent.get_new_thread()
            input_message = ChatMessage(role="user", text=message)
            options = {
                "max_completion_tokens": max_tokens or self.max_completion_tokens,
                "reasoning_effort": self.reasoning_effort,
            }
            async for chunk in agent.run_stream([input_message], thread=thread, options=options):
                if chunk.text:
                    yield chunk.text

    def _model_to_data(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {key: self._model_to_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._model_to_data(item) for item in value]
        return str(value)

    def _to_json(self, value: Any) -> str:
        return json.dumps(self._model_to_data(value), ensure_ascii=False, default=str, indent=2)

    def _get_field(self, value: Any, field_name: str, default: Any = None) -> Any:
        data = self._model_to_data(value)
        if isinstance(data, dict):
            return data.get(field_name, default)
        return getattr(value, field_name, default)

    def _build_history_context(self) -> str:
        messages = self.conversation.get("messages", []) if self.conversation else []
        history = []
        for message in messages[-self.history_max_messages:]:
            role = message.get("role", "user")
            text = message.get("text") or message.get("content") or ""
            if text:
                history.append(f"{role}: {text}")
        return "\n".join(history) if history else "No prior conversation."

    def _append_conversation_turn(self, user_message: str, assistant_message: str) -> None:
        if "messages" not in self.conversation:
            self.conversation["messages"] = []
        self.conversation["messages"].append({"role": "user", "text": user_message})
        self.conversation["messages"].append({"role": "assistant", "text": assistant_message})

    def _extract_json_object(self, text: str) -> Optional[dict[str, Any]]:
        for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _parse_triage_response(self, response: str) -> dict[str, Any]:
        if self._ANSWERED_MARKER in response:
            answer = response.split(self._ANSWERED_MARKER, 1)[-1].strip(" :\n")
            return {"answered": True, "answer": answer or response.replace(self._ANSWERED_MARKER, "").strip()}

        parsed = self._extract_json_object(response) or {}
        datasource_name = (
            parsed.get("datasource_name")
            or parsed.get("datasource")
            or parsed.get("name")
        )
        datasource_type = parsed.get("datasource_type") or parsed.get("type")

        if not datasource_name:
            name_match = re.search(r"datasource(?:_name)?\s*[:=]\s*['\"]?([^'\"\n,}]+)", response, re.IGNORECASE)
            if name_match:
                datasource_name = name_match.group(1).strip()
        if not datasource_type:
            type_match = re.search(r"(?:datasource_)?type\s*[:=]\s*['\"]?([^'\"\n,}]+)", response, re.IGNORECASE)
            if type_match:
                datasource_type = type_match.group(1).strip()

        if datasource_name:
            datasource_name = str(datasource_name).replace(self._DATASOURCE_MARKER, "").strip(" :\n'\"")
        if datasource_type:
            datasource_type = str(datasource_type).strip(" :\n'\"")

        return {
            "answered": False,
            "datasource_name": datasource_name,
            "datasource_type": datasource_type,
        }

    def _extract_sql_query(self, response: str) -> Optional[str]:
        parsed = self._extract_json_object(response) or {}
        query = parsed.get("sql_query") or parsed.get("query")
        if not query:
            fenced = re.search(r"```sql\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
            if fenced:
                query = fenced.group(1)
        if not query:
            select_match = re.search(r"\bselect\b.*", response, flags=re.IGNORECASE | re.DOTALL)
            if select_match:
                query = select_match.group(0)
        if not query:
            return None
        return str(query).strip().rstrip(";")

    def _selected_table_names(self, table_candidates: Any, all_tables: Any) -> list[str]:
        names: list[str] = []
        candidate_data = self._model_to_data(table_candidates)
        all_table_data = self._model_to_data(all_tables)

        for source in (candidate_data, all_table_data):
            tables = source.get("tables", []) if isinstance(source, dict) else []
            for table in tables:
                if isinstance(table, dict):
                    table_name = table.get("table") or table.get("name")
                else:
                    table_name = getattr(table, "table", None) or getattr(table, "name", None)
                if table_name and table_name not in names:
                    names.append(str(table_name))
                if len(names) >= self._MAX_SCHEMA_TABLES:
                    return names
        return names

    async def _collect_schema_context(self, datasource_name: str, user_message: str) -> dict[str, Any]:
        table_candidates = await self._nl2sql_plugin.tables_retrieval(user_message, datasource=datasource_name)
        all_tables = await self._nl2sql_plugin.get_all_tables_info(datasource_name)
        table_names = self._selected_table_names(table_candidates, all_tables)
        schemas_task = asyncio.gather(
            *(self._nl2sql_plugin.get_schema_info(datasource_name, table_name) for table_name in table_names)
        )
        schemas, similar_queries = await asyncio.gather(
            schemas_task,
            self._nl2sql_plugin.queries_retrieval(user_message, datasource=datasource_name),
        )
        return {
            "table_candidates": table_candidates,
            "all_tables": all_tables,
            "schemas": list(schemas),
            "similar_queries": similar_queries,
        }

    def _triage_instructions(self) -> str:
        return (
            f"{self._triage_prompt}\n\n"
            "Use the supplied datasource and table metadata instead of calling external tools. "
            "If the question can be answered without SQL, include QUESTION_ANSWERED followed by the answer. "
            "Otherwise choose exactly one datasource and return JSON like "
            '{"datasource_name":"<name>","datasource_type":"sql_endpoint|sql_database|semantic_model"}.'
        )

    def _sql_generation_instructions(self) -> str:
        return (
            f"{self._sqlquery_prompt}\n\n"
            "Use only the provided metadata. Generate one read-only SQL SELECT query for the selected datasource. "
            "Return only JSON with this shape: "
            '{"sql_query":"SELECT ...","reasoning":"brief explanation"}. '
            "Do not wrap the JSON in markdown and do not execute the query yourself."
        )

    def _synthesis_instructions(self) -> str:
        return (
            f"{self._syntetizer_prompt}\n\n"
            "Synthesize a concise user-facing answer from the SQL result below. "
            "Do not include TERMINATE or internal agent markers."
        )

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        flow_start = time.time()
        await self._load_prompts()
        full_response = ""

        try:
            datasources_info, broad_table_candidates = await asyncio.gather(
                self._nl2sql_plugin.get_all_datasources_info(),
                self._nl2sql_plugin.tables_retrieval(user_message),
            )
            triage_context = (
                f"Conversation history:\n{self._build_history_context()}\n\n"
                f"User question:\n{user_message}\n\n"
                f"Available datasources:\n{self._to_json(datasources_info)}\n\n"
                f"Potentially relevant tables:\n{self._to_json(broad_table_candidates)}"
            )
            triage_response = await self._run_agent(
                self._triage_instructions(),
                triage_context,
                max_tokens=1000,
            )
            selection = self._parse_triage_response(triage_response)

            if selection.get("answered"):
                full_response = str(selection.get("answer") or "").strip()
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            datasource_name = selection.get("datasource_name")
            datasource_type = selection.get("datasource_type")
            if not datasource_name:
                full_response = "I could not identify a configured SQL datasource that matches this question."
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            if datasource_type not in {"sql_endpoint", "sql_database", None}:
                full_response = (
                    f"The selected datasource '{datasource_name}' is type '{datasource_type}', "
                    "which is not currently supported by the SQL execution path."
                )
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            schema_context = await self._collect_schema_context(datasource_name, user_message)
            sql_context = (
                f"Conversation history:\n{self._build_history_context()}\n\n"
                f"User question:\n{user_message}\n\n"
                f"Selected datasource:\n{json.dumps({'name': datasource_name, 'type': datasource_type}, indent=2)}\n\n"
                f"All tables:\n{self._to_json(schema_context['all_tables'])}\n\n"
                f"Relevant tables:\n{self._to_json(schema_context['table_candidates'])}\n\n"
                f"Schemas:\n{self._to_json(schema_context['schemas'])}\n\n"
                f"Similar historical queries:\n{self._to_json(schema_context['similar_queries'])}"
            )
            sql_response = await self._run_agent(
                self._sql_generation_instructions(),
                sql_context,
                max_tokens=1800,
            )
            sql_query = self._extract_sql_query(sql_response)
            if not sql_query:
                full_response = "I could not generate a valid SQL query for this question."
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            validation = await self._nl2sql_plugin.validate_sql_query(sql_query)
            if not self._get_field(validation, "is_valid", False):
                error = self._get_field(validation, "error", "unknown validation error")
                full_response = f"I generated a SQL query, but it did not pass validation: {error}"
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            query_result = await self._nl2sql_plugin.execute_sql_query(datasource_name, sql_query)
            result_error = self._get_field(query_result, "error")
            if result_error:
                full_response = f"The SQL query could not be executed: {result_error}"
                yield full_response
                self._append_conversation_turn(user_message, full_response)
                return

            synthesis_context = (
                f"User question:\n{user_message}\n\n"
                f"Datasource:\n{datasource_name}\n\n"
                f"SQL query:\n{sql_query}\n\n"
                f"SQL result:\n{self._to_json(query_result)}"
            )
            async for chunk in self._stream_agent(
                self._synthesis_instructions(),
                synthesis_context,
                max_tokens=self.max_completion_tokens,
            ):
                full_response += chunk
                yield chunk

            self._append_conversation_turn(user_message, full_response)
            logging.info("[NL2SQLStrategy] Flow completed in %.2fs", time.time() - flow_start)
        except Exception as exc:
            logging.error("[NL2SQLStrategy] Flow failed: %s", exc, exc_info=True)
            full_response = f"I encountered an error processing your NL2SQL request: {exc}"
            yield full_response
            self._append_conversation_turn(user_message, full_response)
