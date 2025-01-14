from tavily import TavilyClient
import os

# conduct tavily search 
def conduct_tavily_search_news(query: str) -> str:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = tavily_client.search(query=query, 
                                   search_depth="advanced",
                                   max_results=2,
                                   topic="news",
                                   days=30) # get news from the last 30 days only 
    return response

def conduct_tavily_search_general(query: str) -> str:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = tavily_client.search(query=query, 
                                   search_depth="advanced",
                                   max_results=2,
                                   topic="general",
                                   days=180) # extended search for up to 180 days
    return response


# format tavily results
def format_tavily_results(response: dict) -> str:
    """
    Format Tavily search results into a clean, readable string.
    
    Args:
        response (dict): Tavily API response dictionary
        
    Returns:
        str: Formatted string of search results
    """
    try:
        if not response.get('results'):
            return "No search results found."
            
        formatted_parts = []
        
        # Add header with search query if present
        if response.get('query'):
            formatted_parts.append(f"Search Query: {response['query']}\n")
            
        # Format each search result
        for i, result in enumerate(response['results'], 1):
            # Format date if available
            date_str = ""
            if result.get('published_date'):
                try:
                    from datetime import datetime
                    date = datetime.strptime(result['published_date'], '%a, %d %b %Y %H:%M:%S %Z')
                    date_str = date.strftime('%B %d, %Y')
                except:
                    date_str = result['published_date']
            
            # Build the result string
            result_str = (
                f"Result {i}:\n"
                f"{'═' * 80}\n"  # Top border
                f"Title: {result.get('title', 'No title')}\n"
                f"Date: {date_str}\n"
                f"Citation: {result.get('url', 'No URL')}\n"
                f"{'─' * 80}\n"  # Separator
                f"{result.get('content', 'No content available')}\n"
                f"{'═' * 80}\n"  # Bottom border
            )
            formatted_parts.append(result_str)
            
        return "\n".join(formatted_parts)
        
    except Exception as e:
        print(f"Error formatting Tavily results: {str(e)}")
        return "Error: Could not format search results"


