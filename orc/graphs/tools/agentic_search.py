from azure.search.documents.indexes.models import (
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentRequestLimits,
    AzureOpenAIVectorizerParameters,
)
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
import os
import json
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Optional, Any


class AgenticSearchConfig:
    """Configuration class for agentic search functionality."""
    
    def __init__(
        self,
        agent_name: str = "rag-agent",
        index_name: str = "ragindex-test",
        azure_openai_endpoint: Optional[str] = None,
        azure_search_endpoint: Optional[str] = None,
        azure_openai_gpt_deployment: str = "Agent",
        azure_openai_gpt_model: str = "gpt-4o",
        credential: Optional[Any] = None,
        reranker_threshold: float = 2.4,
    ):
        self.agent_name = agent_name
        self.index_name = index_name
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_search_endpoint = azure_search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_openai_gpt_deployment = azure_openai_gpt_deployment
        self.azure_openai_gpt_model = azure_openai_gpt_model
        self.credential = credential or DefaultAzureCredential()
        self.reranker_threshold = reranker_threshold
        # Validate required configuration
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be provided")
        if not self.azure_search_endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT must be provided")


class AgenticSearchManager:
    """Main class for managing agentic search operations."""
    
    def __init__(self, config: AgenticSearchConfig):
        """
        Initialize the AgenticSearchManager with configuration.
        
        Args:
            config: AgenticSearchConfig instance with all necessary parameters
        """
        self.config = config
        self._agent_client: Optional[KnowledgeAgentRetrievalClient] = None
        self._search_client: Optional[SearchClient] = None
        self._agent_initialized = False
    
    def agent_exists(self) -> bool:
        """
        Check if the knowledge agent exists without creating it.
        
        Returns:
            bool: True if agent exists, False otherwise
        """
        try:
            index_client = SearchIndexClient(
                endpoint=self.config.azure_search_endpoint, 
                credential=self.config.credential
            )
            
            # Try to get the agent
            agent = index_client.get_agent(self.config.agent_name)
            return agent is not None
            
        except Exception as e:
            # Agent doesn't exist or other error
            print(f"Agent check failed: {str(e)}")
            return False

    def create_or_update_knowledge_agent(self) -> bool:
        """
        Create or update a knowledge agent.
        
        Returns:
            bool: True if successful, False otherwise
        """
        agent = KnowledgeAgent(
            name=self.config.agent_name,
            models=[
                KnowledgeAgentAzureOpenAIModel(
                    azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.config.azure_openai_endpoint,
                        deployment_name=self.config.azure_openai_gpt_deployment,
                        model_name=self.config.azure_openai_gpt_model,
                    )
                )
            ],
            target_indexes=[
                KnowledgeAgentTargetIndex(
                    index_name=self.config.index_name, 
                    default_reranker_threshold=self.config.reranker_threshold
                )
            ],
            request_limits=KnowledgeAgentRequestLimits(max_output_size=10000),
        )

        index_client = SearchIndexClient(
            endpoint=self.config.azure_search_endpoint, 
            credential=self.config.credential
        )
        
        try:
            index_client.create_or_update_agent(agent)
            print(f"Knowledge agent '{self.config.agent_name}' created or updated successfully")
            self._agent_initialized = True
            return True
        except Exception as e:
            print(f"Error creating or updating knowledge agent: {str(e)}")
            return False

    def get_agent_client(self) -> KnowledgeAgentRetrievalClient:
        """
        Get or create the agent client.
        
        Returns:
            KnowledgeAgentRetrievalClient: The initialized agent client
        """
        if self._agent_client is None:
            self._agent_client = KnowledgeAgentRetrievalClient(
                endpoint=self.config.azure_search_endpoint,
                agent_name=self.config.agent_name,
                credential=self.config.credential,
            )
        return self._agent_client

    def get_search_client(self) -> SearchClient:
        """
        Get or create the search client.
        
        Returns:
            SearchClient: The initialized search client
        """
        if self._search_client is None:
            self._search_client = SearchClient(
                endpoint=self.config.azure_search_endpoint,
                index_name=self.config.index_name,
                credential=self.config.credential,
            )
        return self._search_client

    def retrieve_document_by_id(self, document_id: str) -> Optional[Dict]:
        """
        Retrieve a specific document from Azure Search by its ID.
        
        Args:
            document_id (str): The ID of the document to retrieve
            
        Returns:
            dict: The document data or None if not found
            
        Raises:
            Exception: If there's an error during the search operation
        """
        try:
            search_client = self.get_search_client()
            
            # Define the fields to select
            select_fields = [
                "id",
                "content", 
                "title",
                "url",
                "organization_id",
                "date_last_modified"
            ]
            
            # Perform the search with filter by ID
            search_results = search_client.search(
                search_text="*",
                select=select_fields,
                filter=f"id eq '{document_id}'",
                top=1
            )
            
            # Convert search results to list and return first result
            results = list(search_results)
            
            if results:
                return dict(results[0])
            else:
                print(f"No document found with ID: {document_id}")
                return None
                
        except Exception as e:
            print(f"Error retrieving document by ID '{document_id}': {str(e)}")
            raise e

    def enrich_agentic_search_results(self, agentic_results) -> List[Dict]:
        """
        Enrich agentic search results with document metadata.
        
        Args:
            agentic_results: The result object from agentic_retrieval containing references
            
        Returns:
            list: List of enriched results with additional document metadata
        """
        enriched_results = []
        
        try:
            # Convert agentic results to dictionaries if not already
            if hasattr(agentic_results, 'references'):
                references = [r.as_dict() for r in agentic_results.references]
            else:
                references = agentic_results
                
            for result in references:
                # Create a copy of the original result
                enriched_result = result.copy()
                
                # Extract the document ID from source_data
                document_id = None
                if 'source_data' in result and 'id' in result['source_data']:
                    document_id = result['source_data']['id']
                elif 'doc_key' in result:
                    document_id = result['doc_key']
                    
                if document_id:
                    # Retrieve additional document details
                    detailed_doc = self.retrieve_document_by_id(document_id)
                    
                    if detailed_doc:
                        # Add enrichment data as a new section
                        enriched_result['enriched_data'] = {
                            'url': detailed_doc.get('url'),
                            'date_last_modified': detailed_doc.get('date_last_modified'),
                            'organization_id': detailed_doc.get('organization_id'),
                            'content_length': len(detailed_doc.get('content', ''))
                        }
                    else:
                        # Add indication that enrichment failed
                        enriched_result['enriched_data'] = {
                            'error': f'Could not retrieve detailed data for document ID: {document_id}'
                        }
                else:
                    enriched_result['enriched_data'] = {
                        'error': 'No document ID found for enrichment'
                    }
                    
                enriched_results.append(enriched_result)
                
        except Exception as e:
            print(f"Error enriching agentic search results: {str(e)}")
            # Return original results if enrichment fails
            return references if 'references' in locals() else []
            
        return enriched_results

    def agentic_retriever(
        self, 
        conversation_history: List[Dict], 
        reranker_threshold: float = None,
        max_docs_for_reranker: int = 150,
        auto_create_agent: bool = True
    ):
        """
        Perform agentic retrieval based on conversation history.
        Automatically creates/updates the agent if retrieval fails and auto_create_agent is True.
        
        Args:
            conversation_history: list of dictionaries with role and content
            reranker_threshold: threshold for reranker (uses config default if None)
            max_docs_for_reranker: maximum documents for reranker (default: 150)
            auto_create_agent: whether to auto-create agent on failure (default: True)
            
        Returns:
            KnowledgeAgentRetrievalRequest object
        """
        # Use config default if not specified
        if reranker_threshold is None:
            reranker_threshold = self.config.reranker_threshold
            
        try:
            agent_client = self.get_agent_client()
            
            # Convert to KnowledgeAgentMessage format
            agent_messages = agent_client.retrieve(
                retrieval_request=KnowledgeAgentRetrievalRequest(
                    messages=[
                        KnowledgeAgentMessage(
                            role=msg["role"],
                            content=[KnowledgeAgentMessageTextContent(text=msg["content"])],
                        )
                        for msg in conversation_history
                    ],
                    target_index_params=[
                        KnowledgeAgentIndexParams(
                            index_name=self.config.index_name, 
                            reranker_threshold=reranker_threshold, 
                            max_docs_for_reranker=max_docs_for_reranker, 
                            include_reference_source_data=True
                        )
                    ],
                )
            )

            return agent_messages
            
        except Exception as e:
            error_message = str(e).lower()
            
            # Check if the error is related to missing/invalid agent
            agent_related_errors = [
                "not found", "does not exist", "invalid agent", 
                "agent not found", "unauthorized", "forbidden"
            ]
            
            is_agent_error = any(error_text in error_message for error_text in agent_related_errors)
            
            if auto_create_agent and is_agent_error:
                print(f"Agent-related error detected: {str(e)}")
                print("Attempting to create/update the knowledge agent...")
                
                # Try to create or update the agent
                if self.create_or_update_knowledge_agent():
                    print("Agent created/updated successfully. Retrying retrieval...")
                    
                    # Reset the agent client to use the new agent
                    self._agent_client = None
                    
                    # Retry the retrieval
                    try:
                        agent_client = self.get_agent_client()
                        agent_messages = agent_client.retrieve(
                            retrieval_request=KnowledgeAgentRetrievalRequest(
                                messages=[
                                    KnowledgeAgentMessage(
                                        role=msg["role"],
                                        content=[KnowledgeAgentMessageTextContent(text=msg["content"])],
                                    )
                                    for msg in conversation_history
                                ],
                                target_index_params=[
                                    KnowledgeAgentIndexParams(
                                        index_name=self.config.index_name, 
                                        reranker_threshold=reranker_threshold, 
                                        max_docs_for_reranker=max_docs_for_reranker, 
                                        include_reference_source_data=True
                                    )
                                ],
                            )
                        )
                        return agent_messages
                        
                    except Exception as retry_error:
                        print(f"Retry failed after agent creation: {str(retry_error)}")
                        raise retry_error
                else:
                    print("Failed to create/update agent")
                    raise e
            else:
                # Re-raise the original error if it's not agent-related or auto_create is disabled
                print(f"Non-agent error or auto-creation disabled: {str(e)}")
                raise e

    def enrich_and_display_results(self, agentic_results, max_results: int = 5) -> None:
        """
        Enrich agentic search results and display them in a formatted way.
        
        Args:
            agentic_results: The result object from agentic_retrieval
            max_results (int): Maximum number of results to display (default: 5)
        """
        enriched_results = self.enrich_agentic_search_results(agentic_results)
        
        # Create a mapping of activity IDs to search queries
        activity_query_map = {}
        if hasattr(agentic_results, 'activity'):
            for activity in agentic_results.activity:
                activity_dict = activity.as_dict()
                if activity_dict.get("type") == "AzureSearchQuery":
                    activity_id = activity_dict.get("id")
                    search_query = activity_dict.get("query", {}).get("search", "N/A")
                    activity_query_map[activity_id] = search_query
        
        print(f"\n{'='*80}")
        print(f"ENRICHED AGENTIC SEARCH RESULTS (showing {min(len(enriched_results), max_results)} of {len(enriched_results)})")
        print(f"{'='*80}")
        
        for i, result in enumerate(enriched_results[:max_results]):
            print(f"\n--- Result {i+1} ---")
            
            # Basic info
            print(f"Document ID: {result.get('source_data', {}).get('id', 'N/A')}")
            print(f"Title: {result.get('source_data', {}).get('title', 'N/A')}")
            
            # Display search query 
            activity_source_id = result.get('activity_source', 'N/A')
            if activity_source_id != 'N/A' and activity_source_id in activity_query_map:
                search_query = activity_query_map[activity_source_id]
                print(f"Search Query: '{search_query}'")
            else:
                print(f"Activity Source: {activity_source_id}")
            
            # Enriched data
            if 'enriched_data' in result:
                enriched = result['enriched_data']
                if 'error' not in enriched:
                    print(f"URL: {enriched.get('url', 'N/A')}")
                    print(f"Last Modified: {enriched.get('date_last_modified', 'N/A')}")
                    print(f"Organization ID: {enriched.get('organization_id', 'N/A')}")
                else:
                    print(f"Enrichment Error: {enriched['error']}")
            
            # Content preview
            content = result.get('source_data', {}).get('content', '')
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"Content Preview: {preview}")
            
            print("-" * 40)


def create_default_agentic_search_manager() -> AgenticSearchManager:
    """
    Create an AgenticSearchManager with default configuration.
    
    Returns:
        AgenticSearchManager: Configured manager instance
    """
    config = AgenticSearchConfig()
    return AgenticSearchManager(config)


# Test/example code - only runs when script is executed directly
if __name__ == "__main__":
    print("Testing Agentic Search functionality")
    print("=" * 50)
    
    # Create manager with default configuration
    manager = create_default_agentic_search_manager()
    
    # Test conversation
    simple_messages = [
        {"role": "user", "content": "What is marketing?"},
        {"role": "assistant", "content": "Marketing is..."},
    ]
    # add the latest message to the conversation history
    latest_message = {"role": "user", "content": "What is consumer segmentation?"}
    simple_messages.append(latest_message)

    # print the conversation history
    print(f"Conversation history: {simple_messages}")

    try:
        # Perform agentic retrieval (will auto-create agent if needed)
        print("\nPerforming agentic retrieval...")
        agent_messages_simple = manager.agentic_retriever(simple_messages)
        
        print("\n" + "="*50)
        print("Testing agentic search result enrichment")
        print("="*50)
        
        # Enrich and display the agentic search results
        manager.enrich_and_display_results(agent_messages_simple, max_results=1000)
        
        # If you want just the enriched data without display
        enriched_data = manager.enrich_agentic_search_results(agent_messages_simple)
        print(f"\nEnriched {len(enriched_data)} results with additional metadata")
        
    except Exception as e:
        print(f"Agentic search failed: {str(e)}")
        print("This might indicate a configuration issue or service unavailability.")
