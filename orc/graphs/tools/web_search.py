import os
from typing import Dict, List
from langchain.schema import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from shared.util import get_secret

# obtain google search api key
GOOGLE_SEARCH_API_KEY = os.environ.get("SERPER_API_KEY") or get_secret("GoogleSearchKey")

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