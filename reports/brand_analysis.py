import logging
from typing import Optional, Dict, Any
from shared.prompts import henkel_brand_analysis_prompt
from deepagents import create_deep_agent
from reports.config import AgentConfig, EnvironmentConfigLoader

logger = logging.getLogger(__name__)


class BrandAnalysisAgent:
    def __init__(self, config: AgentConfig, brand_prompt: Optional[str] = None, recursion_limit: int = 100) -> None:
        """Initialize the brand analysis agent.

        Args:
            config: Agent configuration containing model and tool settings
            brand_prompt: Optional custom brand analysis prompt
            recursion_limit: Maximum number of recursive calls to the agent
        """
        self.config = config
        self.brand_prompt = brand_prompt or henkel_brand_analysis_prompt
        self.recursion_limit = recursion_limit

    def _init_deep_agent(self):
        """Initialize the deep agent with configured subagents and tools."""
        research_subagent = self.config.init_research_subagent(
            tools=["internet_search"]
        )
        critique_subagent = self.config.init_critique_subagent()
        model = self.config.model_client
        internet_search = self.config.internet_search

        deep_agent = create_deep_agent(
            [internet_search],
            self.brand_prompt,
            subagents=[research_subagent, critique_subagent],
            model=model,
        ).with_config({"recursion_limit": self.recursion_limit})

        return deep_agent

    def generate_report(self, query: str) -> Dict[str, Any]:
        """Generate a brand analysis report.

        Args:
            query: The analysis query string

        Returns:
            Dictionary containing the analysis report

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If report generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        agent = self._init_deep_agent()
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            if not result:
                raise RuntimeError("Agent returned empty result")
            return result
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        config = EnvironmentConfigLoader.load_from_environment()
        agent = BrandAnalysisAgent(config=config, brand_prompt=henkel_brand_analysis_prompt, recursion_limit=1000)
        report = agent.generate_report(
            query="Please provide an analysis of the construction adhesive market in the US"
        )

        if report and isinstance(report, dict) and "files" in report:
            print(report["files"])
        else:
            print("No files found in report or unexpected report format")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
