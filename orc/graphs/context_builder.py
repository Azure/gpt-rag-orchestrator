import logging
from typing import Any, List, Optional
import json

from orc.graphs.utils import clean_chat_history_for_llm
from orc.graphs.constants import (
    TOOL_AGENTIC_SEARCH,
    TOOL_DATA_ANALYST,
    TOOL_DOCUMENT_CHAT,
    TOOL_WEB_FETCH,
)


logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds contextual prompts and converts tool results into context docs."""

    def __init__(self, organization_data: Optional[dict] = None) -> None:
        self.organization_data = organization_data or {}

    def _get_org_data(self, key: str, label: str) -> str:
        value = self.organization_data.get(key, "")
        logger.info(f"[Org Data] Retrieved '{label}' (present={bool(value)})")
        return value

    def build_organization_context_prompt(self, history: List[dict]) -> str:
        """Build the organization context prompt including history and org info."""
        return f"""
        <-------------------------------->
        
        Historical Conversation Context:
        <-------------------------------->
        ```
        {clean_chat_history_for_llm(history)}
        ```
        <-------------------------------->

        **Alias segment mappings:**
        <-------------------------------->
        alias to segment mappings typically look like this (Official Name -> Alias):
        A -> B
        
        This mapping is mostly used in consumer segmentation context. 
        
        Critical Rule – Contextual Consistency with Alias Mapping:
    •\tAlways check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
    •\tALWAYS use the official name (A) in the rewritten query.
    •\tDO NOT use the alias (B) in the rewritten query. 

        Here is the actual alias to segment mappings:
        
        **Official Segment Name Mappings (Official Name -> Alias):**
        ```
        {self._get_org_data('segmentSynonyms', 'Segment Synonyms')}
        ```

        For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

        Look, if a mapping in the instruction is like this:
        students -> young kids 

        Though the historical conversation and the original question may mention "students", you must rewrite the question to use "young kids" instead of "students".

        <-------------------------------->
        Brand Information:
        <-------------------------------->
        ```
        {self._get_org_data('brandInformation', 'Brand Information')}
        ```
        <-------------------------------->

        Industry Information:
        <-------------------------------->
        ```
        {self._get_org_data('industryInformation', 'Industry Information')}
        ```
        <-------------------------------->

        """

    def to_context_docs(self, state) -> List[Any]:
        """
        Convert tool results into plain context fragments for the answerer.

        - agentic_search: include 'results' or the dict as-is
        - data_analyst: include the last agent message and first blob path (if present)
        - web_fetch: include 'content' or the dict as-is
        """
        docs: List[Any] = []

        if not (state.mcp_tool_used and state.tool_results):
            return docs

        for i, tool_call in enumerate(state.mcp_tool_used):
            if i >= len(state.tool_results):
                continue
            result = state.tool_results[i]

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    pass

            name = tool_call.get("name")
            if name == TOOL_AGENTIC_SEARCH and isinstance(result, dict):
                docs.append(result.get("results", result))

            elif name == TOOL_DATA_ANALYST and isinstance(result, dict):
                docs.append(result.get("last_agent_message", result))
                blob_urls = result.get("blob_urls")
                if (
                    isinstance(blob_urls, list)
                    and blob_urls
                    and isinstance(blob_urls[0], dict)
                ):
                    blob_path = blob_urls[0].get("blob_path")
                    if blob_path:
                        logger.info(f"[MCP] Adding blob path to context: {blob_path}")
                        docs.append(
                            f"Here is the graph/visualization link - NEVER CHANGE THE NAME OF THE IMAGE/LINK - THIS IS ABSOLUTELY CRITICAL: \n\n{blob_path}"
                        )

            elif name == TOOL_WEB_FETCH and isinstance(result, dict):
                web_content = result.get("content")
                docs.append(web_content if web_content else result)

            elif name == TOOL_DOCUMENT_CHAT and isinstance(result, dict):
                docs.append(result.get("answer", result))
                files = result.get("files", [])
                if files and isinstance(files, list):
                    state.uploaded_file_refs = files
                    logger.info(
                        f"[MCP] Updated uploaded_file_refs with {len(files)} files"
                    )

        return docs
