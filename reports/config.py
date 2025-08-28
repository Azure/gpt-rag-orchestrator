import os
import logging
from functools import cached_property
from dataclasses import dataclass
from typing import Literal, Union, Optional, get_args
from deepagents import SubAgent
from tavily import TavilyClient
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from shared.prompts import sub_research_prompt, sub_critique_prompt
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

ModelType = Literal["o4-mini", "claude-sonnet-4-20250514", "gpt-4.1"]
ReasoningEffort = Literal["low", "medium", "high"]
ModelClient = Union[AzureChatOpenAI, ChatAnthropic]

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete.

    This exception is raised when required environment variables are missing,
    credentials are invalid, or model configuration is incorrect.
    """


class EnvironmentConfigLoader:
    """Factory class for loading configuration from environment variables."""

    @staticmethod
    def load_from_environment() -> "AgentConfig":
        """Load configuration from environment variables with validation."""
        required_vars = {
            "TAVILY_API_KEY": "tavily_key",
            "AGENT_ENDPOINT_SERVICE": "reasoning_endpoint_service",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "BRAND_ANALYSIS_MODEL": "analysis_model",
        }

        config_dict = {}
        missing_vars = []

        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing_vars.append(env_var)
            else:
                config_dict[config_key] = value

        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # optional config
        config_dict["reasoning_effort"] = os.getenv(
            "REASONING_EFFORT", "high"
        )  # only applicable for o4-mini model

        return AgentConfig(**config_dict)


@dataclass
class AgentConfig:
    """Configuration class for brand analysis agents with lazy-loaded clients."""

    # Configuration constants
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TIMEOUT = 60
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS_ANTHROPIC = 32000
    DEFAULT_MAX_TOKENS = 32000
    DEFAULT_API_VERSION = "2025-04-01-preview"

    tavily_key: str
    reasoning_endpoint_service: str
    anthropic_api_key: str
    analysis_model: ModelType
    reasoning_effort: ReasoningEffort

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_credentials()
        self._validate_model_config()

    def _validate_credentials(self) -> None:
        """Validate that all required credentials are provided."""
        required_fields = [
            "tavily_key",
            "reasoning_endpoint_service",
            "anthropic_api_key",
            "analysis_model",
        ]
        missing_fields = [
            field for field in required_fields if not getattr(self, field)
        ]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required credentials: {', '.join(missing_fields)}"
            )

    @staticmethod
    def _get_token_provider():
        """Get a token provider for the Azure OpenAI client."""
        return get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

    def _validate_model_config(self) -> None:
        """Validate model configuration parameters."""
        # Use type literals for validation
        valid_models = list(get_args(ModelType))
        valid_efforts = list(get_args(ReasoningEffort))

        if self.analysis_model not in valid_models:
            raise ConfigurationError(
                f"Invalid model: {self.analysis_model}. "
                f"Must be one of: {', '.join(valid_models)}"
            )

        if self.reasoning_effort not in valid_efforts:
            raise ConfigurationError(
                f"Invalid reasoning effort: {self.reasoning_effort}. "
                f"Must be one of: {', '.join(valid_efforts)}"
            )

    @cached_property
    def tavily_client(self) -> TavilyClient:
        """Lazy-loaded Tavily client with error handling."""
        try:
            return TavilyClient(api_key=self.tavily_key)
        except Exception:
            logger.error("Failed to initialize Tavily client")
            raise ConfigurationError(
                "Failed to initialize Tavily client. Check API key and connectivity."
            )

    @cached_property
    def model_client(self) -> ModelClient:
        """Lazy-loaded model client with error handling."""
        try:
            if self.analysis_model == "o4-mini":
                return self._create_reasoning_client()
            elif self.analysis_model == "gpt-4.1":
                return self._create_non_reasoning_client()
            else:
                return self._create_anthropic_client()
        except Exception:
            logger.error("Failed to initialize model client")
            raise ConfigurationError(
                "Failed to initialize model client. Check configuration and credentials."
            )

    def _create_reasoning_client(self) -> AzureChatOpenAI:
        """Create and configure Azure OpenAI client."""
        token_provider = self._get_token_provider()
        openai_base_url = f"https://{self.reasoning_endpoint_service}.openai.azure.com/"
        return AzureChatOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=openai_base_url,
            azure_deployment=self.analysis_model,
            max_retries=self.DEFAULT_MAX_RETRIES,
            reasoning_effort=self.reasoning_effort,
            timeout=self.DEFAULT_TIMEOUT,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            api_version=self.DEFAULT_API_VERSION,
        )

    # create a gpt-4.1 client: for quick local testing only - not for production
    def _create_non_reasoning_client(self) -> AzureChatOpenAI:
        """Create and configure GPT-4.1 client."""
        token_provider = self._get_token_provider()
        endpoint = os.getenv("O1_ENDPOINT")

        return AzureChatOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4.1",
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            max_retries=self.DEFAULT_MAX_RETRIES,
            timeout=self.DEFAULT_TIMEOUT,
            api_version=self.DEFAULT_API_VERSION,
        )

    def _create_anthropic_client(self) -> ChatAnthropic:
        """Create and configure Anthropic client."""
        return ChatAnthropic(
            api_key=self.anthropic_api_key,
            model=self.analysis_model,
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS_ANTHROPIC,
            max_retries=self.DEFAULT_MAX_RETRIES,
        )

    def internet_search(self, query: str, include_domains: list[str] = None) -> list[dict]:
        """Run a internet search

        Args:
            query: Search query string
            include_domains: List of domains to include in the search

        Returns:
            List of search results with title, url, and content

        Raises:
            ConfigurationError: If Tavily client initialization failed
        """
        try:
            results = self.tavily_client.search(
                query,
                max_results=3,
                include_domains=include_domains,
                include_raw_content=False,
                topic="general",
                time_range="week",
                search_depth="advanced",
            )
            return results
        except Exception as e:
            logger.error(f"Failed to run web search: {str(e)}")
            return []

    def init_research_subagent(self, tools: list[str]) -> dict:
        """Initialize a research subagent.

        Args:
            tools: List of tool names available to the subagent

        Returns:
            Dictionary containing research subagent configuration
        """
        return {
            "name": "research-agent",
            "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
            "prompt": sub_research_prompt,
            "tools": tools,
            # "model": model # TODO: config a model for this sub agent so that it won't use the same model as the main agent
        }

    def init_critique_subagent(self) -> dict:
        """Initialize a critique subagent.

        Returns:
            Dictionary containing critique subagent configuration
        """
        return {
            "name": "critique-agent",
            "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
            "prompt": sub_critique_prompt,
        }
