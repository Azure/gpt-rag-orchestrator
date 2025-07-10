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

# class QueryPlanner(BaseModel):
#     """
#     You are an expert query planning assistant for an AI system. Your purpose is to
#     break down a user's request into three distinct, actionable sub-queries based on the 'How, Why, and What' framework:
#     1. `strategy_query`: The 'How' (Process & Frameworks).
#     2. `consumer_insight_query`: The 'Why' (Human Motivation & Psychology).
#     3. `marketing_query`: The 'What' (Market & Product Facts).
#     """
    
#     original_query: str = Field(description="The original query from the user.")
#     # sub_query: SubQuery = Field(..., description="The generated Strategy, Consumer Insight, and Marketing sub-queries.")
#     sub_queries: list[str] = Field(..., description=""" 
#     The generated Strategy, Consumer Insight, and Marketing sub-queries.
#     The sub-queries should be a list of strings, each string is a sub-query.
#     The sub-queries should be a list of strings, each string is a sub-query.
#     """)

class QueryPlanner(BaseModel):
    """
    You are an expert query decomposition for a Retrieval-Augmented Generation (RAG) system. Your purpose is to break the query down into a set of subqueries that have clear, complete, fully qualified, concise, and self-contained propositions in order to optimize database search.
    However, if the user's query is simple, straightforward, and does not require any decomposition, you should only generate one sub-query.
    The sub-queries should be a list of strings, each string is a sub-query.
    Express each idea or fact as a standalone statement, that can be understood with the help of the given historical conversation, the original query, and the rewritten query. 
    You need to identify the main elements of the sentence, typically a subject, an action or relationship, and an object or complement. Determine which element is being asked about or emphasized (usually the unknown or focus of the question). Invert the sentence structure. Make the original object or complement the new subject. Transform the original subject into a descriptor or qualifier. Adjust the verb or relationship to fit the new structure.
    Make sure you separate complex ideas into multiple sub simpler propositions when appropriate
    Decontextualize each proposition by adding necessary modifiers to nouns or entire sentences. Replace pronouns, such as it, he, she, they, this, and that, with the full name of the entities that they refer to.
    Each sub-query should be brief, a database search friendly on sentence length, and should be a statement, not a question.
    Each sub-query should capture distinct aspects of the original query and should not be too similar to each other.
    
    Note: The rewritten query tends to be more marketing focus, which sometimes may be completely unrelated to the original query. In this case, you should rely on the original query and the historical conversation to generate sub-queries to gain more context to generate more relevant sub-queries.
    
    Here are some examples of how to decompose a query into sub-queries:
    1. Query: What is the capital of France?
    Sub-queries:
    - What is the capital of France?

    2. Query: "Who is the current CEO of the company that created the iPhone?"
    Sub-queries:
    - Company created the iPhone
    - Current CEO of Apple
    
    3. Query: How can we improve brand loyalty for our premium coffee subscription service?
    Sub-queries:
    - Coffee subscription service loyalty drivers
    - Premium brand loyalty building strategies
    - Coffee consumer segmentation and behavior
    - Subscription business retention best practices

    4. Query: What advertising channels should we use to reach Gen Z consumers for our new sustainable fashion line?
    Sub-queries:
    - Gen Z media consumption habits and platform preferences
    - Sustainable fashion advertising best practices
    - Gen Z values and purchase drivers for fashion
    - Digital advertising ROI by channel for fashion brands

    5. Rewritten query: "Top competitors of marketing agencies in Raleigh, North Carolina" <-- This rewritten query was clearly unrelated to the original query, and it would be very bad to generate sub-queries based on this rewritten query.
    Original query: "What are the competitors of the company?" <-- This is the raw/original query from the user which may be lacking context.
    Historical conversation: "The conversation was about the user wants to open a new hair salon in Raleigh, North Carolina. " <-- This historical conversation provides additional context what user was discussing before.
    Sub-queries:
    - Top competitors of hair salons in Raleigh, North Carolina
    - Hair salon industry trends in Raleigh, North Carolina
    - Hair salon consumer segmentation and behavior 
    - Hair salon business retention best practices 
    - How to stand out in the hair salon industry 


    """

    original_query: str = Field(description="The original query from the user.")
    sub_queries: list[str] = Field(..., description="The generated sub-queries. Should be no more than 4 sub-queries.")

# Model configuration for query planning
model = AzureChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    max_tokens=5000,
    api_key=os.getenv("O1_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("O1_ENDPOINT"),
)

def generate_sub_queries(original_query: str, rewritten_query: str, historical_conversation: str, model: AzureChatOpenAI) -> list[str]:
    # TODO: fall back strategy if model fails to generate sub-queries
    prompt = f"""
    Original query: {original_query}
    Rewritten query: {rewritten_query}
    Historical conversation: {historical_conversation}
    """
    query_planner = model.with_structured_output(QueryPlanner).invoke(prompt)
    return query_planner.sub_queries


if __name__ == "__main__":
    # Template for the creative brief - example usage
    queries = generate_sub_queries("Write a marketing plan for a new product of Primo Water", "Write a marketing plan for a new product of Primo Water", "nothing in the historical conversation yet", model)
    query_list = [query.strip() for query in queries]
    print(query_list)
