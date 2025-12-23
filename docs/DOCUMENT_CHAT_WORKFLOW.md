# Document Chat Workflow

## Overview
This document explains how the document chat workflow is implemented end to end. Users can attach up to three documents in the UI; the backend then routes the query to the document chat MCP tool, reuses prior file IDs when possible, and persists state to Cosmos DB for continuity across turns.

## End‑to‑End Flow
1. UI sends `question`, `conversation_id`, and `blob_names` (array of document blob names).
2. Function app extracts `blob_names` and passes them to the orchestrator.
3. Orchestrator normalizes `blob_names` and builds the conversation graph.
4. Graph routes to tools immediately when `blob_names` are present in the runtime state (passed via `ainvoke`).
5. MCP executor forces the `document_chat` tool, validates cache vs. UI input, and calls the tool.
6. Context builder adds the tool’s `answer` to context docs and overwrites `uploaded_file_refs` in state.
7. Graph returns state; orchestrator saves `uploaded_file_refs` and other metadata to Cosmos.

## Storage Layout (UI/ingestion expectation)
```
container_name/
  organization_id/
    user_id/
      conversation_id/
        file_1_id
        file_2_id
        file_3_id   # UI enforces max 3
```

## MCP Contract

Request arguments to `document_chat` (sent by MCP executor):
```python
{
    "question": state.question,
    "document_names": state.blob_names,            # always present when tool is used
    # only present if validation passes
    "cached_file_info": state.uploaded_file_refs,
}
```

Response shape expected from `document_chat`:
```python
{
    "answer": "...",
    "files": [
        {"file_id": "file-abc123", "blob_name": "doc1.pdf"},
        {"file_id": "file-def456", "blob_name": "doc2.pdf"}
    ],
    "metadata": { ... }
}
```

## Implementation Map (current code)

- Conversation state
  - `orc/graphs/models.py`: `ConversationState` includes `blob_names: List[str]` and `uploaded_file_refs: List[Dict[str, str]]`.

- HTTP entrypoint
  - `function_app.py`: `/api/orc` extracts `blob_names` from JSON and passes them to the orchestrator (a small default list is present for local testing).

- Orchestrator
  - `orc/new_orchestrator.py`:
    - `generate_response_with_progress(...)` accepts and normalizes `blob_names`.
    - Passes `blob_names` only in the `agent.ainvoke({"question", "blob_names"}, ...)` payload; the graph uses this to seed `state.blob_names`.
    - Builds `ConversationState` from the graph’s result and includes `uploaded_file_refs`.
    - Appends `uploaded_file_refs` and `last_mcp_tool_used` to the assistant message in Cosmos history.

- Graph and routing
  - `orc/graphs/main_2.py`:
    - `GraphBuilder` no longer accepts `blob_names`.
    - `_route_decision(...)` returns `tool_choice` when `state.blob_names` (from `ainvoke` input) is present.
    - `_return_state(...)` includes `uploaded_file_refs` in the returned dict.

- History initialization
  - `orc/graphs/query_planner.py` uses utilities from `orc/graphs/utils.py` to extract `code_thread_id`, `last_mcp_tool_used`, and `uploaded_file_refs` from Cosmos history during rewrite; these seed the state at the start of each turn.

- MCP tool planning and execution
  - `orc/graphs/mcp_executor.py`:
    - `get_tool_calls(...)` forces `document_chat` if `state.blob_names` is non‑empty (skips LLM tool planning).
    - `_configure_document_chat_args(...)` compares `state.blob_names` with `state.uploaded_file_refs[*].blob_name`:
      - Exact set match → include `cached_file_info` for file ID reuse.
      - Mismatch or empty cache → omit `cached_file_info` to process fresh documents.
    - When `state.blob_names` is empty, `document_chat` is excluded from the set of tools bound to the LLM during planning to avoid invalid schema binding; the LLM will only choose among other available tools.

- Context building
  - `orc/graphs/context_builder.py`: for `document_chat`, appends `result["answer"]` to context docs and overwrites `state.uploaded_file_refs` with `result["files"]`.

## Validation and Reuse Rules
- Exact set match on blob names (case‑sensitive) controls reuse; order is ignored.
- No partial reuse — any difference triggers fresh processing.
- Always overwrite `uploaded_file_refs` with the latest `files` from the tool.
- UI enforces up to 3 files; backend accepts any length but uses what the UI provides.

## Data Flow Summary

1) Request path
```
UI → Function App → Orchestrator → GraphBuilder → ConversationState (with `blob_names` provided via `ainvoke`)
```

2) Validation path
```
state.blob_names  vs  state.uploaded_file_refs[*].blob_name  →  cached_file_info
```

3) Response path
```
MCP Response {answer, files} → ContextBuilder → state.uploaded_file_refs overwrite → _return_state → Cosmos
```

## Example Payloads

- HTTP request to `/api/orc` (body excerpt):
```json
{
  "question": "Summarize the contract differences",
  "conversation_id": "<uuid>",
  "blob_names": ["doc1.pdf", "doc2.pdf"]
}
```

- Tool args sent to MCP `document_chat` on a cache hit:
```json
{
  "question": "...question...",
  "document_names": ["doc1.pdf", "doc2.pdf"],
  "cached_file_info": [
    {"file_id": "file-abc123", "blob_name": "doc1.pdf"},
    {"file_id": "file-def456", "blob_name": "doc2.pdf"}
  ]
}
```

## Testing Scenarios

1) First upload (no cache)
- Input: `blob_names=["doc1.pdf"]`, history contains no `uploaded_file_refs`.
- Expect: send only `document_names`; tool returns `files`; state overwrites `uploaded_file_refs`.

2) Same documents (cache hit)
- Input: `blob_names=["doc1.pdf"]`, history `uploaded_file_refs=[{"file_id":"abc","blob_name":"doc1.pdf"}]`.
- Expect: include `cached_file_info`; tool reuses file ID; state overwrites with returned `files`.

3) Different documents (cache miss)
- Input: `blob_names=["doc2.pdf"]`, history refs for `doc1.pdf`.
- Expect: omit `cached_file_info`; process fresh; overwrite with new refs.

4) Expired files (tool handles)
- Input: cache hit scenario but IDs expired on MCP side.
- Expect: server detects expiration; falls back to fresh processing; returns new IDs; overwrite state.

## Notes and Limitations
- `function_app.py` includes a small default `blob_names` list for local testing; clients should always pass their own array.
- The 3‑file limit is enforced in the UI; backend logic does not hard‑enforce it.
- Blob name comparison is case‑sensitive to avoid ambiguous reuse.

## Status
The document chat workflow described above is fully implemented and live in the codebase outlined in the Implementation Map.
