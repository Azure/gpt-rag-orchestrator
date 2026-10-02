"""
Legacy profile models and serialization compatibility.

Automatic profile context and extraction are suspended under ADR-0006 until
trusted key ownership and collection authorization have been established.
"""

import asyncio
import logging
from collections.abc import MutableSequence, Sequence
from typing import Any, Optional, List

from pydantic import BaseModel, Field
from agent_framework import ContextProvider, Context, ChatClientProtocol, ChatMessage




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
    Compatibility profile container with explicitly disabled framework hooks.
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
        """Collection is explicitly disabled for every adapter and caller."""
        logging.debug("profile_extraction_disabled")

    async def _extract_and_update_profile(self, messages_list: list[ChatMessage]) -> None:
        """Compatibility entrypoint: never send messages to an extraction model."""
        logging.debug("profile_extraction_disabled")

    async def flush(self) -> None:
        """No extraction work is scheduled or invented by flushing."""
        logging.debug("profile_extraction_disabled")

    async def invoking(
        self,
        _messages: ChatMessage | MutableSequence[ChatMessage],
        **_kwargs: Any
    ) -> Context:
        """Do not inject a legacy profile without verified owner binding."""
        logging.debug("profile_access_disabled_unverified_binding")
        return Context()

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
