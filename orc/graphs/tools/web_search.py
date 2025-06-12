import os
from typing import Dict, List
from langchain.schema import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from shared.util import get_secret
# import sys
from tavily import TavilyClient

# # get the path of the parent directory
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # add the parent directory to the system path
# sys.path.append(parent_dir)

# obtain google search api key
GOOGLE_SEARCH_API_KEY = os.environ.get("SERPER_API_KEY") or get_secret("GoogleSearchKey")

# obtain tavily api key
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY") or get_secret("TavilyAPIKey")

class GoogleSearch:
    """
    A class to handle Google search operations using the Serper API
    """
    def __init__(self, k: int = 3):
        self.search_tool = GoogleSerperAPIWrapper(
            k=k, 
            serper_api_key=GOOGLE_SEARCH_API_KEY
        )

    def ggsearch_reformat(self, result: Dict) -> List[Document]:
        """
        Reformats Google search results into a list of Document objects.

        Args:
            result (Dict): The raw search results from Google.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """
        documents = []
        try:
            # Process Knowledge Graph results if present
            if 'knowledgeGraph' in result:
                kg = result['knowledgeGraph']
                documents.append(Document(
                    page_content=kg.get('description', ''),
                    metadata={
                        'source': kg.get('descriptionLink', ''), 
                        'title': kg.get('title', '')
                    }
                ))
            
            # Process organic search results
            documents.extend([
                Document(
                    page_content=item.get('snippet', ''),
                    metadata={
                        'source': item.get('link', ''), 
                        'title': item.get('title', '')
                    }
                )
                for item in result.get('organic', [])
            ])
            
            if not documents:
                raise ValueError("No search results found")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            documents.append(Document(
                page_content="No search results found or an error occurred.",
                metadata={'source': 'Error', 'title': 'Search Error'}
            ))
        
        return documents

    def search(self, query: str) -> List[Document]:
        """
        Performs a Google search and returns formatted results.

        Args:
            query (str): The search query

        Returns:
            List[Document]: A list of Document objects containing the search results
        """
        results = self.search_tool.results(query)
        return self.ggsearch_reformat(results)

class TavilySearch:
    """
    A class to handle web search operations using the Tavily API
    """
    def __init__(self, max_results: int = 3, search_depth: str = "advanced"):
        """
        Initialize TavilySearch with configuration parameters.
        
        Args:
            max_results (int): Maximum number of search results to return (default: 3)
            search_depth (str): Search depth - "basic" or "advanced" (default: "advanced")
        """
        self.max_results = max_results
        self.search_depth = search_depth
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def tavily_reformat(self, result: Dict) -> List[Document]:
        """
        Reformats Tavily search results into a list of Document objects.

        Args:
            result (Dict): The raw search results from Tavily.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """
        documents = []
        try:
            # Process search results
            for item in result.get('results', []):
                # Create document with content and metadata
                documents.append(Document(
                    page_content=item.get('content', ''),
                    metadata={
                        'source': item.get('url', ''),
                        'title': item.get('title', '')
                    }
                ))
            
            if not documents:
                raise ValueError("No search results found")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            documents.append(Document(
                page_content="No search results found or an error occurred.",
                metadata={'source': 'Error', 'title': 'Search Error'}
            ))
        
        return documents

    def search_news(self, query: str, days: int = 30) -> List[Document]:
        """
        Performs a Tavily news search and returns formatted results.

        Args:
            query (str): The search query
            days (int): Number of days back to search for news (default: 30)

        Returns:
            List[Document]: A list of Document objects containing the search results
        """
        try:
            results = self.client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                topic="news",
                days=days
            )
            return self.tavily_reformat(results)
        except Exception as e:
            print(f"Error in Tavily news search: {str(e)}")
            return [Document(
                page_content="Error occurred during news search.",
                metadata={'source': 'Error', 'title': 'Search Error'}
            )]

    def search_general(self, query: str, days: int = 180) -> List[Document]:
        """
        Performs a Tavily general search and returns formatted results.

        Args:
            query (str): The search query
            days (int): Number of days back to search (default: 180)

        Returns:
            List[Document]: A list of Document objects containing the search results
        """
        try:
            results = self.client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                topic="general",
                days=days
            )
            return self.tavily_reformat(results)
        except Exception as e:
            print(f"Error in Tavily general search: {str(e)}")
            return [Document(
                page_content="Error occurred during general search.",
                metadata={'source': 'Error', 'title': 'Search Error'}
            )]

    def search(self, query: str, topic: str = "general", days: int = 360) -> List[Document]:
        """
        Performs a Tavily search and returns formatted results.
        This is the main search method that provides a unified interface.

        Args:
            query (str): The search query
            topic (str): Search topic - "general" or "news" (default: "general")
            days (int): Number of days back to search (default: 180)

        Returns:
            List[Document]: A list of Document objects containing the search results
        """
        if topic == "news":
            return self.search_news(query, days)
        else:
            return self.search_general(query, days)

if __name__ == "__main__":
    search = TavilySearch()
    results = search.search("What is the capital of France?")
    print(results)