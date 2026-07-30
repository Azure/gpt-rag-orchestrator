"""Translate Microsoft Agent Framework updates into neutral turn events."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from agent_framework import (
    CitationAnnotation,
    FunctionCallContent,
    FunctionResultContent,
)

from orchestration.turn import (
    TurnCitation,
    TurnCitationEvent,
    TurnToolActivity,
    TurnToolActivityEvent,
    TurnToolStatus,
)


class AgentEventTranslator:
    """Stateful translator that correlates streamed tool calls and results."""

    def __init__(self) -> None:
        self._tool_names: dict[str, str] = {}
        self._started_calls: set[str] = set()
        self._citation_index = 0

    def translate(
        self,
        update: Any,
    ) -> Iterator[TurnCitationEvent | TurnToolActivityEvent]:
        for content in getattr(update, "contents", ()):
            if isinstance(content, FunctionCallContent):
                if content.call_id and content.call_id not in self._started_calls:
                    self._started_calls.add(content.call_id)
                    self._tool_names[content.call_id] = content.name
                    yield TurnToolActivityEvent(
                        activity=TurnToolActivity(
                            tool_name=content.name,
                            status=TurnToolStatus.STARTED,
                            call_id=content.call_id,
                        )
                    )
            elif isinstance(content, FunctionResultContent):
                tool_name = self._tool_names.pop(content.call_id, "unknown")
                self._started_calls.discard(content.call_id)
                yield TurnToolActivityEvent(
                    activity=TurnToolActivity(
                        tool_name=tool_name,
                        status=(
                            TurnToolStatus.FAILED
                            if content.exception is not None
                            else TurnToolStatus.COMPLETED
                        ),
                        call_id=content.call_id,
                        message=(
                            "Tool execution failed"
                            if content.exception is not None
                            else None
                        ),
                    )
                )
            yield from self._citation_events(
                getattr(content, "annotations", None)
            )

    def _citation_events(
        self,
        annotations: Any,
    ) -> Iterator[TurnCitationEvent]:
        for annotation in annotations or ():
            values = self._citation_values(annotation)
            if values is None:
                continue
            self._citation_index += 1
            citation_id = (
                values.get("file_id")
                or values.get("url")
                or f"citation-{self._citation_index}"
            )
            yield TurnCitationEvent(
                citation=TurnCitation(
                    citation_id=str(citation_id),
                    title=values.get("title"),
                    url=values.get("url"),
                    snippet=values.get("snippet"),
                )
            )

    @staticmethod
    def _citation_values(annotation: Any) -> dict[str, Any] | None:
        if isinstance(annotation, CitationAnnotation):
            return {
                "file_id": annotation.file_id,
                "title": annotation.title,
                "url": annotation.url,
                "snippet": annotation.snippet,
            }
        if isinstance(annotation, Mapping):
            return {
                "file_id": annotation.get("file_id"),
                "title": annotation.get("title"),
                "url": annotation.get("url"),
                "snippet": annotation.get("snippet"),
            }
        return None
