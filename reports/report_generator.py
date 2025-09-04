import logging
from typing import Dict, Any
from functools import cached_property

# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)
from shared.prompts import (
    competitor_analysis_prompt,
    product_analysis_prompt,
    brand_analysis_prompt,
)
from deepagents import create_deep_agent
from reports.config import AgentConfig, EnvironmentConfigLoader
from shared.util import normalize_markdown

logger = logging.getLogger(__name__)

# Mapping of report types to their corresponding prompts
REPORT_TYPE_PROMPTS = {
    "brand_analysis": brand_analysis_prompt,
    "product_analysis": product_analysis_prompt,
    "competitor_analysis": competitor_analysis_prompt,
}


class ReportGenerator:
    def __init__(
        self, config: AgentConfig, agent_prompt: str, recursion_limit: int = 100
    ) -> None:
        """Initialize the report generator.

        Args:
            config: Agent configuration containing model and tool settings
            agent_prompt: custom agent prompt
            recursion_limit: Maximum number of recursive calls to the agent
        """
        self.config = config
        self.agent_prompt = agent_prompt
        self.recursion_limit = recursion_limit

    @classmethod
    def from_report_type(
        cls, config: AgentConfig, report_type: str, recursion_limit: int = 100
    ) -> "ReportGenerator":
        """Create a ReportGenerator instance from a specific report type.

        Args:
            config: Agent configuration
            report_type: Type of report to generate
            recursion_limit: Maximum recursion limit

        Returns:
            ReportGenerator instance configured for the report type

        Raises:
            ValueError: If report_type is invalid
        """
        if report_type not in REPORT_TYPE_PROMPTS:
            raise ValueError(
                f"Invalid report_type: {report_type}. Must be one of {list(REPORT_TYPE_PROMPTS.keys())}"
            )

        prompt = REPORT_TYPE_PROMPTS.get(report_type)
        return cls(config=config, agent_prompt=prompt, recursion_limit=recursion_limit)

    @cached_property
    def agent(self):
        """Lazy-loaded deep agent instance."""
        research_subagent = self.config.init_research_subagent(
            tools=["internet_search"]
        )
        critique_subagent = self.config.init_critique_subagent()
        model = self.config.model_client
        internet_search = self.config.internet_search

        return create_deep_agent(
            [internet_search],
            self.agent_prompt,
            subagents=[research_subagent, critique_subagent],
            model=model,
        ).with_config({"recursion_limit": self.recursion_limit})

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

        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            if not result:
                raise RuntimeError("Agent returned empty result")
            return result
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}", exc_info=True)
            raise

    def format_report_output(self, report: dict) -> str:
        """Format the report output to be a markdown string.

        Args:
            report: Dictionary containing the report files

        Returns:
            Formatted markdown report content
        """
        report_content = normalize_markdown(report.get("final_report.md", ""))
        return report_content

    def generate_and_format_report(self, query: str) -> str:
        """Generate a report and return formatted markdown content.

        Args:
            query: The analysis query string

        Returns:
            Formatted markdown report content

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If report generation fails or no files found
        """
        report = self.generate_report(query)

        if report and isinstance(report, dict) and "files" in report:
            return self.format_report_output(report["files"])
        else:
            raise RuntimeError("No files found in report or unexpected report format")


# [START] this is for langgraph studio only
report_agent = ReportGenerator(
    config=EnvironmentConfigLoader.load_from_environment(),
    agent_prompt=brand_analysis_prompt,
    recursion_limit=1000,
).agent
# [END]


def run_analysis(query: str, report_type: str, recursion_limit: int = 100) -> str:
    """Run analysis with the given query and return formatted report.

    Args:
        query: The analysis query string
        report_type: Type of report to generate
        recursion_limit: Maximum recursion limit for the agent

    Returns:
        Formatted markdown report content

    Raises:
        ValueError: If report_type is invalid or query is empty
        RuntimeError: If report generation fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        config = EnvironmentConfigLoader.load_from_environment()
        generator = ReportGenerator.from_report_type(
            config=config,
            report_type=report_type,
            recursion_limit=recursion_limit,
        )
        return generator.generate_and_format_report(query=query)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise RuntimeError(f"Analysis failed: {e}") from e


if __name__ == "__main__":
    competitor_analysis_query = "Please provide a 2-page competitor analysis on these brands: sika, 3m, bostik. The industry is construction adhesives and sealants."
    product_analysis_query = """
    Please generate a monthly product performance report for the following consumer adhesive and sealant products:

    Super Glue Gel Control - Super Glue

    Silicone 1 Tub & Tile - Caulk & Sealants

    Gaps & Cracks Insulating Foam - Foam Insulation

    Silicone 1 All Purpose Sealant Window & Door - Caulk & Sealants

    Advanced Silicone Kitchen & Bath Sealant - Caulk & Sealants

    PL Premium Polyurethane Construction Adhesive - Construction Adhesive

    Power Grab Mounting Tape - Mounting Tape
"""
    brand_analysis_query = """ 
    Please generate the weekly Brand Analysis Report.

    Brand Focus: Loctite's consumer and construction adhesives business.

    Industry Context: Consumer & Professional Adhesives and Sealants.
"""
    try:
        result = run_analysis(
            query=product_analysis_query,
            report_type="product_analysis",
            recursion_limit=100,
        )
        print(result)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
