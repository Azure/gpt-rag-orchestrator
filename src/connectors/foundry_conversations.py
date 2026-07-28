"""Managed Microsoft Foundry Conversations operations."""

from azure.ai.projects.aio import AIProjectClient


async def resolve_managed_conversation_id(
    project_client: AIProjectClient,
    requested_id: str | None,
) -> str:
    """Validate an existing managed Conversation or create a new one."""
    normalized_id = requested_id.strip() if requested_id else None
    if requested_id is not None and not normalized_id:
        raise ValueError("Foundry conversation_id must not be blank.")

    async with project_client.get_openai_client() as openai_client:
        if normalized_id:
            conversation = await openai_client.conversations.retrieve(normalized_id)
        else:
            conversation = await openai_client.conversations.create()

    conversation_id = getattr(conversation, "id", None)
    if not isinstance(conversation_id, str) or not conversation_id:
        raise RuntimeError(
            "Foundry Conversations API returned a conversation without a valid id."
        )
    if normalized_id and conversation_id != normalized_id:
        raise RuntimeError(
            "Foundry Conversations API returned a different conversation id "
            f"while validating '{normalized_id}'."
        )
    return conversation_id
