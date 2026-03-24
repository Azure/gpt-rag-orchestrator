"""
User Profile Memory - Context provider for maintaining user profile information.

This module provides a reusable context provider that:
- Extracts user information from conversations
- Persists user profile across sessions
- Provides user context to guide agent responses
"""

import asyncio
import logging
from collections.abc import MutableSequence, Sequence
from typing import Any, Optional, List

from pydantic import BaseModel, Field
from agent_framework import ContextProvider, Context, ChatClientProtocol, ChatMessage, ChatOptions




# ============================================================================
# Pydantic Models for Structured Memory
# ============================================================================

class UserProfile(BaseModel):
    """User profile - persisted across sessions."""
    name: Optional[str] = Field(default=None, description="User's name")
    role: Optional[str] = Field(default=None, description="User's role/title")
    company: Optional[str] = Field(default=None, description="User's company name")
    preferences: List[str] = Field(default_factory=list, description="User preferences")
    notes: List[str] = Field(default_factory=list, description="Additional notes about the user")


class ExtractedUserInfo(BaseModel):
    """Structured extraction of user profile information from conversation."""
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ============================================================================
# Memory Context Provider
# ============================================================================

class UserProfileMemory(ContextProvider):
    """
    Context provider that maintains the user's profile.

    Extracts user information from conversations and provides context
    about the user to guide the agent's responses.
    """

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        user_profile: Optional[UserProfile] = None,
        **kwargs: Any
    ):
        self._chat_client = chat_client
        self._pending_task: Optional[asyncio.Task] = None
        if user_profile:
            self.user_profile = user_profile
        elif kwargs:
            self.user_profile = UserProfile.model_validate(kwargs)
        else:
            self.user_profile = UserProfile()

    def has_minimum_context(self) -> bool:
        """Check if we have minimum required user profile information."""
        return bool(self.user_profile.name)

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        _response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        _invoke_exception: Exception | None = None,
        **_kwargs: Any,
    ) -> None:
        """Schedule profile extraction as a background task (non-blocking)."""
        messages_list = [request_messages] if isinstance(request_messages, ChatMessage) else list(request_messages)
        user_messages = [msg for msg in messages_list if msg.role.value == "user"]

        if not user_messages:
            return

        if self._chat_client.__class__.__name__ == "AzureAIAgentClient":
            logging.debug("[UserProfileMemory] Skipping profile extraction for AzureAIAgentClient")
            return

        # Cancel any previous pending extraction
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

        self._pending_task = asyncio.create_task(self._extract_and_update_profile(messages_list))

    async def _extract_and_update_profile(self, messages_list: list[ChatMessage]) -> None:
        """Extract user profile information from messages (runs as background task)."""
        try:
            result = await self._chat_client.get_response(
                messages=messages_list,
                chat_options=ChatOptions(
                    instructions=(
                        "Extract any information about the USER from the conversation. "
                        "Look for: their name, role/title, company, preferences, and any other relevant notes. "
                        "Only extract information that is explicitly stated about the user. "
                        "Return nulls/empty lists for fields not mentioned."
                    ),
                    response_format=ExtractedUserInfo,
                ),
            )

            if result.value and isinstance(result.value, ExtractedUserInfo):
                extracted = result.value
                if extracted.name:
                    self.user_profile.name = extracted.name
                if extracted.role:
                    self.user_profile.role = extracted.role
                if extracted.company:
                    self.user_profile.company = extracted.company
                if extracted.preferences:
                    self.user_profile.preferences.extend(
                        p for p in extracted.preferences if p not in self.user_profile.preferences
                    )
                if extracted.notes:
                    self.user_profile.notes.extend(
                        n for n in extracted.notes if n not in self.user_profile.notes
                    )

                logging.debug(f"[UserProfileMemory] Updated user profile: {self.user_profile}")

        except asyncio.CancelledError:
            logging.debug("[UserProfileMemory] Profile extraction cancelled")
        except Exception as e:
            if isinstance(e, AttributeError) and "conversation_id" in str(e):
                logging.debug(f"[UserProfileMemory] Skipped unsupported extraction path: {e}")
            else:
                logging.warning(f"[UserProfileMemory] Failed to extract user info: {e}")

    async def flush(self) -> None:
        """Await any pending profile extraction task. Call before saving the profile."""
        if self._pending_task and not self._pending_task.done():
            try:
                await self._pending_task
            except Exception:
                pass  # Already logged in _extract_and_update_profile
        self._pending_task = None

    async def invoking(
        self,
        _messages: ChatMessage | MutableSequence[ChatMessage],
        **_kwargs: Any
    ) -> Context:
        """Provide user profile context before each agent call."""
        instructions: List[str] = []

        if self.has_minimum_context():
            profile_summary = self._build_profile_summary()
            instructions.append(f"User Profile:\n{profile_summary}")

        return Context(instructions="\n".join(instructions))

    def _build_profile_summary(self) -> str:
        """Build a formatted summary of the user profile."""
        parts = []
        p = self.user_profile

        if p.name:
            parts.append(f"- Name: {p.name}")
        if p.role:
            parts.append(f"- Role: {p.role}")
        if p.company:
            parts.append(f"- Company: {p.company}")
        if p.preferences:
            parts.append(f"- Preferences: {', '.join(p.preferences)}")
        if p.notes:
            parts.append(f"- Notes: {'; '.join(p.notes[:3])}")  # Limit to 3

        return "\n".join(parts) if parts else "No user profile information available."

    def serialize(self) -> str:
        """Serialize the user profile for persistence."""
        return self.user_profile.model_dump_json()

    @classmethod
    def deserialize(cls, data: str, chat_client: ChatClientProtocol) -> "UserProfileMemory":
        """Deserialize a user profile from stored data."""
        profile = UserProfile.model_validate_json(data)
        return cls(chat_client=chat_client, user_profile=profile)
