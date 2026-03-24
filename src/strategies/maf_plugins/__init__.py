"""
MAF Plugins - Extensible components for the Microsoft Agent Framework strategy.

This package contains reusable context providers and memory components
that can be used with the MafStrategy.
"""

from .user_profile_memory import (
    UserProfile,
    ExtractedUserInfo,
    UserProfileMemory,
)

__all__ = [
    "UserProfile",
    "ExtractedUserInfo",
    "UserProfileMemory",
]
