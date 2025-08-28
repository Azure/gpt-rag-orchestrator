import logging
from typing import Optional, Dict, Any
from functools import cached_property
from shared.prompts import henkel_brand_analysis_prompt, competitor_analysis_prompt
from deepagents import create_deep_agent
from reports.config import AgentConfig, EnvironmentConfigLoader

logger = logging.getLogger(__name__)


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
            result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
            if not result:
                raise RuntimeError("Agent returned empty result")
            return result
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}", exc_info=True)
            raise


def run_analysis(query: str, prompt: str, recursion_limit: int = 500) -> None:
    """Run brand analysis with the given query."""
    try:
        config = EnvironmentConfigLoader.load_from_environment()
        agent = ReportGenerator(
            config=config,
            agent_prompt=prompt,
            recursion_limit=recursion_limit,
        )
        report = agent.generate_report(query=query)

        if report and isinstance(report, dict) and "files" in report:
            return report["files"]
        else:
            print("No files found in report or unexpected report format")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")

def normalize_markdown(md_raw: str) -> str:
    import html, re
    # If double-encoded (starts/ends with quotes), unquote once
    if md_raw and md_raw[0] in "\"'" and md_raw[-1] == md_raw[0]:
        try:
            import json
            md_raw = json.loads(md_raw)
        except Exception:
            pass
    md = md_raw.replace("\r\n", "\n").replace("\r", "\n")
    md = (md
          .replace("\\r", "\r")
          .replace("\\n", "\n")
          .replace("\\t", "\t"))
    md = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), md)
    md = html.unescape(md)
    # Optional spacing guards omitted for brevity
    return md.strip()

if __name__ == "__main__":
    result = run_analysis(
        query="Please provide a 2-page competitor analysis on these brands: sika, 3m, bostik. The industry is construction adhesives and sealants.",
        prompt=competitor_analysis_prompt,
        recursion_limit=1000,
    )
    if result:
        report_content = normalize_markdown(result.get("final_report.md", ""))
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        print("Report saved to report.md")
    else:
        print("No result found")
