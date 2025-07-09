import os
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI


class SubQuery(BaseModel):
    """
    Defines three distinct sub-queries to help an AI assistant perform a strategic task.
    The queries are structured around the 'How' (Strategy/Process), the 'Why' (Consumer Motivation),
    and the 'What' (Market/Domain Knowledge).
    It should be brief, on sentence length, and should be a statement, not a question.
    """
    
    strategy_query: str = Field(
        description=(
            "The 'How' Query. This query is designed to find the principles and process of creation. "
            "Its goal is to find best practices, frameworks, or expert advice on HOW to perform the core task in the original query. "
            "It must be a statement, not a question."
            "Example: For 'How to write a strategy brief for a new PC', a good strategy query is "
            "'Best practices for writing a compelling and effective strategy brief'."
        )
    )
    
    marketing_query: str = Field(
        description=(
            "The 'What' Query. This query focuses on the specific market facts, domain knowledge, and competitive landscape. "
            "Its goal is to find data and analysis ABOUT the product, industry trends, and audience demographics. "
            "It must be a statement, not a question. "
            "Example: For 'write a strategy brief for a new PC', a good query is "
            "'Market analysis, key competitors, and target demographics for the consumer PC industry'."
        )
    )
    
    consumer_insight_query: str = Field(
        description=(
            "The 'Why' Query. This query is designed to uncover the psychological and emotional drivers of the target consumer. "
            "It focuses on their core motivations, unmet needs, cultural context, and the human truth behind their behavior. "
            "It must be a statement, not a question. "
            "Example: For 'write a strategy brief for a new PC', a good query is "
            "'The role of high-performance technology in personal identity and creative expression'."
        )
    )

class QueryPlanner(BaseModel):
    """
    You are an expert query planning assistant for an AI system. Your purpose is to
    break down a user's request into three distinct, actionable sub-queries based on the 'How, Why, and What' framework:
    1. `strategy_query`: The 'How' (Process & Frameworks).
    2. `consumer_insight_query`: The 'Why' (Human Motivation & Psychology).
    3. `marketing_query`: The 'What' (Market & Product Facts).
    """
    
    original_query: str = Field(description="The original query from the user.")
    sub_query: SubQuery = Field(..., description="The generated Strategy, Consumer Insight, and Marketing sub-queries.")


# Model configuration for query planning
model = AzureChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    max_tokens=5000,
    api_key=os.getenv("O1_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("O1_ENDPOINT"),
)

def generate_sub_queries(original_query: str, model: AzureChatOpenAI) -> SubQuery:
    """
    You are an expert query planning assistant for a Retrieval-Augmented Generation (RAG) system.
    Your goal is to expand a user's original query into three distinct, powerful sub-queries.
    These sub-queries will be used to search a database for relevant information.
    Generate one 'strategy_query' for inspiration, one 'consumer_insight_query' for practical guidance, 
    and one 'marketing_query' for practical guidance.
    """
    # TODO: fall back strategy if model fails to generate sub-queries
    query_planner = model.with_structured_output(QueryPlanner).invoke(original_query)
    return query_planner.sub_query


if __name__ == "__main__":
    # Template for the creative brief - example usage
    query = generate_sub_queries("Write a marketing plan for a new product of Primo Water", model)
    
    print("Strategy Query:", query.strategy_query)
    print("Marketing Query:", query.marketing_query)
    print("Consumer Insight Query:", query.consumer_insight_query) 