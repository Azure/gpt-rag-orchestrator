"""Owner-scoped OData filters for conversation-uploaded retrieval chunks.

A conversation id is chosen by the client and runtime uploads can be indexed
before the conversation document exists, so the id alone is not proof of
ownership. Upload chunks carry the uploader's object id in
``metadata_security_user_ids``; every conversation-scoped retrieval filter must
require that owner in addition to the conversation id.
"""

from typing import Any, Mapping, Optional

ANONYMOUS_PRINCIPAL_ID = "anonymous"
CONVERSATION_FIELD = "conversationId"
CONVERSATION_OWNER_FIELD = "metadata_security_user_ids"


def odata_escape_string(value: Optional[str]) -> str:
    """Escape a string for embedding in single-quoted OData literals."""
    return (value or "").replace("'", "''")


def resolve_conversation_owner_id(user_context: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the server-derived principal that owns conversation uploads.

    ``user_context`` principal keys are populated by the server from the
    validated token (client-supplied values are stripped in ``main.py``), so
    they are safe to use as the upload owner.
    """
    if not isinstance(user_context, Mapping):
        return None
    for key in ("principal_id", "oid"):
        value = str(user_context.get(key) or "").strip()
        if value:
            return value
    return None


def build_conversation_owner_clause(
    conversation_id: Optional[str],
    owner_id: Optional[str],
    *,
    field_name: str = CONVERSATION_FIELD,
    owner_field: str = CONVERSATION_OWNER_FIELD,
) -> Optional[str]:
    """Build the clause matching uploads of ``conversation_id`` owned by ``owner_id``.

    - authenticated owner: only chunks stamped with that object id match;
    - anonymous owner: only chunks without an owner ACL match, so anonymous
      callers can never read an authenticated user's uploads;
    - no owner or no conversation id: ``None`` (no conversation chunks).
    """
    cid = (conversation_id or "").strip()
    owner = (owner_id or "").strip()
    if not cid or not owner:
        return None
    safe_field = (field_name or "").strip() or CONVERSATION_FIELD
    safe_owner_field = (owner_field or "").strip() or CONVERSATION_OWNER_FIELD
    cid_clause = f"{safe_field} eq '{odata_escape_string(cid)}'"
    if owner == ANONYMOUS_PRINCIPAL_ID:
        owner_clause = f"not {safe_owner_field}/any()"
    else:
        owner_clause = f"{safe_owner_field}/any(u: u eq '{odata_escape_string(owner)}')"
    return f"({cid_clause} and {owner_clause})"


def build_shared_conversation_clause(field_name: str = CONVERSATION_FIELD) -> str:
    """Match shared/global chunks (conversationId is the 'NaN' sentinel or null)."""
    safe_field = (field_name or "").strip() or CONVERSATION_FIELD
    return f"({safe_field} eq 'NaN' or {safe_field} eq null)"


def build_conversation_filter(
    conversation_id: Optional[str],
    *,
    owner_id: Optional[str] = None,
    field_name: str = CONVERSATION_FIELD,
    owner_field: str = CONVERSATION_OWNER_FIELD,
) -> str:
    """Build OData filter for conversation-scoped retrieval.

    Includes:
    - conversation-specific chunks (conversationId == <cid>) owned by
      ``owner_id`` when both are set (see ``build_conversation_owner_clause``)
    - shared/global chunks always. A chunk is treated as shared when its
      conversationId is the 'NaN' sentinel OR null/unset. Ingestion has used
      both representations for global corpora, so both must match here,
      otherwise globally-ingested documents become invisible to retrieval.
    """
    safe_field = (field_name or "").strip() or CONVERSATION_FIELD
    shared_clause = build_shared_conversation_clause(safe_field)
    owned_clause = build_conversation_owner_clause(
        conversation_id, owner_id, field_name=safe_field, owner_field=owner_field
    )
    if owned_clause:
        return f"{owned_clause} or {shared_clause}"
    return shared_clause
