"""Registry of orchestrator settings exposed by the admin dashboard.

This module is the **single source of truth** for the curated set of App
Configuration keys the admin Configuration tab can read and write. It owns
the human-readable labels, tooltip descriptions, per-option help text, types,
and validation bounds — both the backend endpoints and the frontend tab
consume the same metadata (the frontend receives it as part of the GET
response so we never duplicate the strings on the client).

Two security primitives live here as well:

* :data:`ALLOWED_KEYS` — the orchestrator-specific allowlist. Only keys in
  this set may ever be returned by ``GET /api/dashboard/config`` or accepted
  by ``PUT /api/dashboard/config``.
* :data:`DENYLIST` — defense in depth: even if a future refactor accidentally
  adds a sensitive key to the registry, writes to anything matching the
  denylist are rejected at the API layer.

Adding a new setting is a one-line change here. Adding a new sensitive key to
deny is also a one-line change. The intent is that this file is reviewed
specifically when settings are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from strategies.agent_strategies import AgentStrategies


SettingType = Literal["enum", "bool", "int", "float"]


@dataclass(frozen=True)
class SettingOption:
    """One entry in an enum dropdown, with per-value tooltip text."""
    value: str
    label: str
    description: str


@dataclass(frozen=True)
class SettingSpec:
    """Metadata for a single exposed App Configuration key."""
    key: str
    type: SettingType
    default: Any
    label: str
    description: str
    options: Optional[List[SettingOption]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: Optional[str] = None


@dataclass(frozen=True)
class SettingSection:
    """A group of related settings rendered as one card in the UI."""
    id: str
    label: str
    description: str
    settings: List[SettingSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enum option helpers
# ---------------------------------------------------------------------------

# The dropdown only exposes the strategies actually documented in issue #512.
# ``AgentStrategies`` also contains ``multiagent`` (legacy) which we
# deliberately omit. Each option's description explains the trade-off so a
# screen reader user gets the same context as a sighted user hovering the (i).
_AGENT_STRATEGY_OPTIONS: List[SettingOption] = [
    SettingOption(
        value=AgentStrategies.SINGLE_AGENT_RAG.value,
        label="Single agent RAG",
        description=(
            "One Semantic Kernel agent answers each turn using the configured "
            "retrieval plugins. Best for classic Retrieval Augmented Generation "
            "use cases with predictable latency and cost."
        ),
    ),
    SettingOption(
        value=AgentStrategies.MAF_LITE.value,
        label="Microsoft Agent Framework (lite)",
        description=(
            "Single MAF chat agent without the Foundry agent service. Lower "
            "latency than the full agent service and the recommended starting "
            "point for new MAF deployments."
        ),
    ),
    SettingOption(
        value=AgentStrategies.MAF_AGENT_SERVICE.value,
        label="Microsoft Agent Framework (Foundry Agent Service)",
        description=(
            "MAF agent backed by the hosted Foundry Agent Service. Adds server "
            "side tool execution and persistent threads at the cost of an extra "
            "network hop per turn."
        ),
    ),
    SettingOption(
        value=AgentStrategies.MCP.value,
        label="MCP",
        description=(
            "Routes tool calls through Model Context Protocol servers. Use when "
            "the orchestrator must expose tools defined by an external MCP host."
        ),
    ),
    SettingOption(
        value=AgentStrategies.NL2SQL.value,
        label="Natural language to SQL",
        description=(
            "Specialised strategy that translates user questions into SQL "
            "against the configured database. Disables free-form RAG behaviour."
        ),
    ),
    SettingOption(
        value=AgentStrategies.MULTIMODAL.value,
        label="Multimodal",
        description=(
            "Image-aware strategy that can classify and reason about images in "
            "the conversation. Pair with MULTIMODAL_CLASSIFY_IMAGES below."
        ),
    ),
]


_REASONING_EFFORT_OPTIONS: List[SettingOption] = [
    SettingOption("minimal", "Minimal", "Fastest and cheapest. Skip almost all reasoning steps."),
    SettingOption("low", "Low", "Some chain-of-thought. Good default for short tasks."),
    SettingOption("medium", "Medium", "Balanced reasoning depth. Recommended default."),
    SettingOption("high", "High", "Deeper deliberation. Higher latency and cost."),
]


# ---------------------------------------------------------------------------
# Sections (renders as one card per section in the UI, top to bottom)
# ---------------------------------------------------------------------------

SECTIONS: List[SettingSection] = [
    SettingSection(
        id="agent_and_generation",
        label="Agent and generation",
        description=(
            "Which orchestration strategy answers each turn and how creative "
            "the underlying chat model is allowed to be."
        ),
        settings=[
            SettingSpec(
                key="AGENT_STRATEGY",
                type="enum",
                default=AgentStrategies.SINGLE_AGENT_RAG.value,
                label="Agent strategy",
                description=(
                    "Selects the orchestration strategy that handles each user "
                    "turn. Changing this swaps which agent class is "
                    "instantiated; new requests pick up the change after the "
                    "configuration cache is refreshed."
                ),
                options=_AGENT_STRATEGY_OPTIONS,
            ),
            SettingSpec(
                key="REASONING_EFFORT",
                type="enum",
                default="medium",
                label="Reasoning effort",
                description=(
                    "Maps to the OpenAI Responses API ``reasoning.effort`` "
                    "parameter. Lower effort is faster and cheaper; higher "
                    "effort lets the model deliberate longer before answering."
                ),
                options=_REASONING_EFFORT_OPTIONS,
            ),
            SettingSpec(
                key="CHAT_TEMPERATURE",
                type="float",
                default=1.0,
                label="Chat temperature",
                description=(
                    "Controls randomness of generated text. 0 is deterministic, "
                    "1 is the model default, values above 1 increase creativity "
                    "but can produce off-topic or hallucinated content."
                ),
                min=0.0,
                max=2.0,
                step=0.1,
            ),
            SettingSpec(
                key="CHAT_TOP_P",
                type="float",
                default=1.0,
                label="Chat top-p (nucleus sampling)",
                description=(
                    "Limits sampling to the smallest set of tokens whose "
                    "cumulative probability is at least top-p. Lower values "
                    "make the model more focused; 1.0 disables nucleus "
                    "filtering."
                ),
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            SettingSpec(
                key="MAX_COMPLETION_TOKENS",
                type="int",
                default=4096,
                label="Max completion tokens",
                description=(
                    "Upper bound on the number of tokens generated per "
                    "response. Larger values allow longer answers but also "
                    "increase latency and cost; the model may still stop "
                    "earlier on its own."
                ),
                min=1,
                max=32768,
                step=1,
                unit="tokens",
            ),
        ],
    ),
    SettingSection(
        id="conversation_history",
        label="Conversation history",
        description=(
            "How much prior conversation is replayed back to the model and "
            "whether older turns are compacted to keep the window small."
        ),
        settings=[
            SettingSpec(
                key="CHAT_HISTORY_MAX_MESSAGES",
                type="int",
                default=20,
                label="Max history messages",
                description=(
                    "Maximum number of prior turns replayed to the model on "
                    "every request. Higher values give the model more context "
                    "but use more tokens per call."
                ),
                min=0,
                max=200,
                step=1,
                unit="messages",
            ),
            SettingSpec(
                key="CONVERSATION_HISTORY_COMPACTION_ENABLED",
                type="bool",
                default=False,
                label="Compact long histories",
                description=(
                    "When enabled, older turns are summarised once the history "
                    "exceeds the max-messages threshold. Reduces token usage on "
                    "long sessions but introduces an extra summarisation call."
                ),
            ),
        ],
    ),
    SettingSection(
        id="retrieval_and_search",
        label="Retrieval and search",
        description=(
            "Which retrieval sources the orchestrator consults before "
            "answering, and how many results each source returns."
        ),
        settings=[
            SettingSpec(
                key="SEARCH_RETRIEVAL_ENABLED",
                type="bool",
                default=True,
                label="Enable Azure AI Search retrieval",
                description=(
                    "Turn the AI Search retrieval plugin on or off globally. "
                    "When off, the orchestrator answers purely from model "
                    "knowledge and any other enabled tools."
                ),
            ),
            SettingSpec(
                key="SEARCH_RAGINDEX_TOP_K",
                type="int",
                default=5,
                label="AI Search top-K",
                description=(
                    "How many documents to retrieve from the RAG index per "
                    "query. Higher values improve recall but increase prompt "
                    "size and latency."
                ),
                min=1,
                max=50,
                step=1,
            ),
            SettingSpec(
                key="BING_RETRIEVAL_ENABLED",
                type="bool",
                default=False,
                label="Enable Bing web grounding",
                description=(
                    "Allow the orchestrator to call the Bing grounding plugin "
                    "for fresh web results. Requires BING_CONNECTION_ID to be "
                    "configured on the Foundry project."
                ),
            ),
        ],
    ),
    SettingSection(
        id="reliability",
        label="Reliability",
        description="How aggressively the orchestrator retries transient model failures.",
        settings=[
            SettingSpec(
                key="INFERENCE_MAX_RETRIES",
                type="int",
                default=3,
                label="Inference max retries",
                description=(
                    "Number of times to retry a failed chat completion call "
                    "before giving up. Higher values improve resilience to "
                    "transient errors at the cost of longer worst-case "
                    "latency."
                ),
                min=0,
                max=10,
                step=1,
            ),
        ],
    ),
    SettingSection(
        id="multimodal",
        label="Multimodal",
        description="Image handling for multimodal strategies.",
        settings=[
            SettingSpec(
                key="MULTIMODAL_CLASSIFY_IMAGES",
                type="bool",
                default=False,
                label="Classify attached images",
                description=(
                    "When enabled, attached images are first sent to a vision "
                    "classifier so downstream agents can react to image "
                    "categories. Disable to skip the extra call when image "
                    "context is not needed."
                ),
            ),
        ],
    ),
]


# Flat allowlist derived from SECTIONS. Anything not in here is invisible to
# the dashboard, both for reads and writes.
ALLOWED_KEYS = frozenset(
    spec.key for section in SECTIONS for spec in section.settings
)


# Explicit denylist of keys that must never be exposed or written through the
# dashboard. The PUT handler enforces this in addition to the allowlist as
# defense-in-depth.
DENYLIST = frozenset(
    {
        "MCP_APP_APIKEY",
        "ORCHESTRATOR_APP_APIKEY",
        "KEY_VAULT_URI",
        "APP_API_TOKEN",
        "DAPR_API_TOKEN",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_SECRET",
        "OAUTH_AZURE_AD_CLIENT_SECRET",
        "OAUTH_AZURE_AD_CLIENT_ID",
        "OAUTH_AZURE_AD_TENANT_ID",
        "APP_CONFIG_ENDPOINT",
    }
)

# Suffix patterns that auto-rejects new sensitive keys without a code change.
_DENY_SUFFIXES = (
    "_APIKEY",
    "_API_KEY",
    "_SECRET",
    "_PASSWORD",
    "_CONNECTION_STRING",
    "_CONNSTRING",
    "_PRIVATE_KEY",
    "_TOKEN",
)

_DENY_PREFIXES = ("OAUTH_",)


def is_denied(key: str) -> bool:
    """Return ``True`` when ``key`` must never be exposed via the dashboard."""
    if key in DENYLIST:
        return True
    upper = key.upper()
    if any(upper.endswith(suffix) for suffix in _DENY_SUFFIXES):
        return True
    if any(upper.startswith(prefix) for prefix in _DENY_PREFIXES):
        return True
    return False


def find_spec(key: str) -> Optional[SettingSpec]:
    """Return the :class:`SettingSpec` for ``key``, or ``None`` if not exposed."""
    for section in SECTIONS:
        for spec in section.settings:
            if spec.key == key:
                return spec
    return None
