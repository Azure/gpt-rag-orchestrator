"""Validated configuration and request credentials for Foundry IQ MCP sources."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_KNOWLEDGE_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_NAME_RE = re.compile(r"^[0-9A-Za-z-]{1,127}$")
_DENIED_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_RESERVED_HOSTS = frozenset({"localhost", "localhost.localdomain"})
_RESERVED_HOST_SUFFIXES = (".localhost", ".local", ".invalid", ".test", ".example")


class McpConfigurationError(ValueError):
    """Raised when enabled MCP configuration cannot be used safely."""


class McpCredentialError(RuntimeError):
    """Raised when a configured MCP query credential cannot be resolved."""


class McpJsonOutputParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    documents_path: str = Field(alias="documentsPath")

    @field_validator("documents_path")
    @classmethod
    def validate_documents_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("documentsPath must not be empty")
        return value


class McpOutputParsing(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["auto", "json", "split", "none"] = "auto"
    json_parameters: Optional[McpJsonOutputParameters] = Field(
        default=None, alias="jsonParameters"
    )
    split_parameters: Optional[dict[str, Any]] = Field(
        default=None, alias="splitParameters"
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "McpOutputParsing":
        if self.kind == "json" and self.json_parameters is None:
            raise ValueError("json outputParsing requires jsonParameters.documentsPath")
        if self.kind != "json" and self.json_parameters is not None:
            raise ValueError("jsonParameters is only valid for json outputParsing")
        if self.kind != "split" and self.split_parameters is not None:
            raise ValueError("splitParameters is only valid for split outputParsing")
        return self


class McpTool(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    max_output_tokens: Optional[int] = Field(default=None, alias="maxOutputTokens")
    inclusion_mode: Optional[Literal["all", "reranked"]] = Field(
        default=None, alias="inclusionMode"
    )
    output_parsing: McpOutputParsing = Field(
        default_factory=McpOutputParsing, alias="outputParsing"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool name must not be empty")
        return value

    @field_validator("max_output_tokens")
    @classmethod
    def validate_max_output_tokens(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not 1 <= value <= 8192:
            raise ValueError("maxOutputTokens must be between 1 and 8192")
        return value


class McpHeaderValueFrom(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["managedIdentity", "obo", "keyVaultSecret", "none"]
    scope: Optional[str] = None
    secret_name: Optional[str] = Field(default=None, alias="secretName")

    @model_validator(mode="after")
    def validate_metadata(self) -> "McpHeaderValueFrom":
        if self.kind in {"managedIdentity", "obo"}:
            if not self.scope or not self.scope.strip():
                raise ValueError(f"{self.kind} requires an explicit scope")
            if any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.scope):
                raise ValueError("scope contains control characters")
            self.scope = self.scope.strip()
        elif self.scope is not None:
            raise ValueError(f"scope is not valid for {self.kind}")

        if self.kind == "keyVaultSecret":
            if not self.secret_name or not _SECRET_NAME_RE.fullmatch(self.secret_name):
                raise ValueError(
                    "keyVaultSecret requires a valid Key Vault secretName"
                )
        elif self.secret_name is not None:
            raise ValueError(f"secretName is not valid for {self.kind}")
        return self


class McpQueryHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    value_from: McpHeaderValueFrom = Field(alias="valueFrom")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not _HEADER_NAME_RE.fullmatch(value):
            raise ValueError("header name is not a valid HTTP field name")
        if value.lower() in _DENIED_HEADERS:
            raise ValueError(f"header {value!r} is not allowed")
        return value


class McpSource(BaseModel):
    """Provisioning metadata plus retrieve-time policy for one MCP source."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    server_url: str = Field(alias="serverUrl")
    fail_on_error: bool = Field(default=True, alias="failOnError")
    max_output_documents: Optional[int] = Field(
        default=None, alias="maxOutputDocuments"
    )
    tools: list[McpTool]
    query_headers: list[McpQueryHeader] = Field(
        default_factory=list, alias="queryHeaders"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not _KNOWLEDGE_SOURCE_NAME_RE.fullmatch(value):
            raise ValueError(
                "source name must use only letters, numbers, '.', '_' or '-'"
            )
        return value

    @field_validator("max_output_documents")
    @classmethod
    def validate_max_output_documents(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not 1 <= value <= 50:
            raise ValueError("maxOutputDocuments must be between 1 and 50")
        return value

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: list[McpTool]) -> list[McpTool]:
        if not value:
            raise ValueError("at least one tool is required")
        names = [tool.name.casefold() for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique within a source")
        return value

    @field_validator("query_headers")
    @classmethod
    def validate_query_headers(
        cls, value: list[McpQueryHeader]
    ) -> list[McpQueryHeader]:
        names = [header.name.casefold() for header in value]
        if len(names) != len(set(names)):
            raise ValueError("query header names must be unique within a source")
        return value


def _parse_trusted_hosts(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return frozenset()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise McpConfigurationError(
                    "FOUNDRY_IQ_MCP_TRUSTED_HOSTS is not valid JSON"
                ) from exc
        else:
            value = re.split(r"[,\r\n]+", text)
    if not isinstance(value, (list, tuple, set)):
        raise McpConfigurationError(
            "FOUNDRY_IQ_MCP_TRUSTED_HOSTS must be a host list"
        )

    hosts: set[str] = set()
    for item in value:
        host = str(item).strip().rstrip(".").lower()
        if not host or "://" in host or "/" in host or ":" in host:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MCP_TRUSTED_HOSTS entries must be hostnames only"
            )
        hosts.add(host)
    return frozenset(hosts)


def _validate_server_url(source: McpSource, trusted_hosts: frozenset[str]) -> None:
    try:
        parsed = urlsplit(source.server_url)
    except ValueError as exc:
        raise McpConfigurationError(
            f"MCP source {source.name!r} has an invalid serverUrl"
        ) from exc

    host = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise McpConfigurationError(
            f"MCP source {source.name!r} serverUrl must be HTTPS without "
            "userinfo or a fragment"
        )
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise McpConfigurationError(
            f"MCP source {source.name!r} serverUrl must not use an IP literal"
        )
    if (
        host in _RESERVED_HOSTS
        or host.endswith(_RESERVED_HOST_SUFFIXES)
        or "." not in host
    ):
        raise McpConfigurationError(
            f"MCP source {source.name!r} serverUrl uses a local or reserved host"
        )
    if host not in trusted_hosts:
        raise McpConfigurationError(
            f"MCP source {source.name!r} host {host!r} is not in "
            "FOUNDRY_IQ_MCP_TRUSTED_HOSTS"
        )


def _format_validation_error(exc: ValidationError) -> str:
    """Return field paths and reasons without Pydantic's rejected input values."""
    messages: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg") or "invalid value")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages) or "invalid source configuration"


@dataclass(frozen=True)
class McpRuntimeConfig:
    enabled: bool
    sources: tuple[McpSource, ...] = ()
    reasoning_effort: Literal["low", "medium"] = "low"
    log_tool_arguments: bool = False

    @classmethod
    def parse(
        cls,
        *,
        enabled: bool,
        sources_json: Any,
        reasoning_effort: Any,
        trusted_hosts: Any,
        log_tool_arguments: bool,
        api_version: str,
        max_runtime_seconds: Any,
    ) -> "McpRuntimeConfig":
        if not enabled:
            return cls(enabled=False)
        if api_version != "2026-05-01-preview":
            raise McpConfigurationError(
                "Foundry IQ MCP sources require API version 2026-05-01-preview"
            )
        try:
            runtime_seconds = int(max_runtime_seconds)
        except (TypeError, ValueError) as exc:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MAX_RUNTIME_SECONDS must be an integer"
            ) from exc
        if isinstance(max_runtime_seconds, bool) or not 30 <= runtime_seconds <= 600:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MAX_RUNTIME_SECONDS must be between 30 and 600 "
                "when MCP sources are enabled"
            )

        reasoning = str(reasoning_effort or "").strip().lower()
        if reasoning not in {"low", "medium"}:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MCP_REASONING_EFFORT must be 'low' or 'medium'"
            )

        trusted = _parse_trusted_hosts(trusted_hosts)
        if not trusted:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MCP_TRUSTED_HOSTS is required when MCP is enabled"
            )

        try:
            raw_sources = (
                json.loads(sources_json) if isinstance(sources_json, str) else sources_json
            )
        except json.JSONDecodeError as exc:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MCP_SOURCES_JSON is not valid JSON"
            ) from exc
        if not isinstance(raw_sources, list) or not raw_sources:
            raise McpConfigurationError(
                "FOUNDRY_IQ_MCP_SOURCES_JSON must contain at least one source"
            )
        try:
            sources = tuple(McpSource.model_validate(item) for item in raw_sources)
        except ValidationError as exc:
            raise McpConfigurationError(
                "Invalid FOUNDRY_IQ_MCP_SOURCES_JSON: "
                f"{_format_validation_error(exc)}"
            ) from None
        except TypeError:
            raise McpConfigurationError(
                "Invalid FOUNDRY_IQ_MCP_SOURCES_JSON source structure"
            ) from None

        names = [source.name.casefold() for source in sources]
        if len(names) != len(set(names)):
            raise McpConfigurationError("MCP source names must be unique")
        for source in sources:
            _validate_server_url(source, trusted)

        return cls(
            enabled=True,
            sources=sources,
            reasoning_effort=reasoning,  # type: ignore[arg-type]
            log_tool_arguments=bool(log_tool_arguments),
        )

    def knowledge_source_params(self) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        for source in self.sources:
            item: dict[str, Any] = {
                "knowledgeSourceName": source.name,
                "kind": "mcpServer",
                "includeReferences": True,
                "includeReferenceSourceData": True,
                "failOnError": source.fail_on_error,
            }
            if source.max_output_documents is not None:
                item["maxOutputDocuments"] = source.max_output_documents
            params.append(item)
        return params

    def source_by_name(self) -> Mapping[str, McpSource]:
        return {source.name: source for source in self.sources}


def _format_token_value(header_name: str, token: str) -> str:
    if header_name.casefold() == "authorization":
        return f"Bearer {token}"
    return token


async def build_mcp_control_headers(
    config: McpRuntimeConfig,
    *,
    credential: Any,
    incoming_token: Optional[str],
    acquire_obo_token: Callable[[str, str], Awaitable[Optional[str]]],
    get_secret: Callable[[str], Awaitable[Optional[str]]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Resolve configured credentials into deterministic paired control headers."""

    headers: dict[str, str] = {}
    credential_modes: dict[str, tuple[str, ...]] = {}

    for source in config.sources:
        resolved: list[tuple[str, str, str]] = []
        for query_header in source.query_headers:
            value_from = query_header.value_from
            if value_from.kind == "none":
                continue
            if value_from.kind == "managedIdentity":
                token = (await credential.get_token(value_from.scope)).token
                if not token:
                    raise McpCredentialError(
                        f"MCP source {source.name!r} managed identity returned no token"
                    )
                value = _format_token_value(query_header.name, token)
            elif value_from.kind == "obo":
                if not incoming_token:
                    raise McpCredentialError(
                        f"MCP source {source.name!r} requires an incoming user token"
                    )
                token = await acquire_obo_token(incoming_token, value_from.scope or "")
                if not token:
                    raise McpCredentialError(
                        f"MCP source {source.name!r} OBO exchange returned no token"
                    )
                value = _format_token_value(query_header.name, token)
            else:
                value = await get_secret(value_from.secret_name or "")
                if not value:
                    raise McpCredentialError(
                        f"MCP source {source.name!r} Key Vault credential is missing"
                    )
            if "\r" in value or "\n" in value:
                raise McpCredentialError(
                    f"MCP source {source.name!r} credential contains control characters"
                )
            resolved.append((query_header.name, value, value_from.kind))

        modes: list[str] = []
        for index, (header_name, header_value, mode) in enumerate(resolved):
            suffix = "" if index == 0 else str(index)
            prefix = f"{source.name}-header"
            headers[f"{prefix}-name{suffix}"] = header_name
            headers[f"{prefix}-value{suffix}"] = header_value
            modes.append(mode)
        credential_modes[source.name] = tuple(modes or ["none"])

    return headers, credential_modes
