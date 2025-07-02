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
import logging
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Optional, Any
from langchain.schema import Document
import sys

# Set up logging for Azure Functions - this needs to be done before creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # Override any existing logging configuration
)

# Configure the main module logger
logger = logging.getLogger(__name__)

# Configure Azure SDK specific loggers as per Azure SDK documentation
# Set logging level for Azure Search libraries
azure_search_logger = logging.getLogger("azure.search")
azure_search_logger.setLevel(logging.INFO)

# Set logging level for Azure Identity libraries
azure_identity_logger = logging.getLogger("azure.identity")
azure_identity_logger.setLevel(logging.WARNING)  # Less verbose for auth

# Set logging level for all Azure libraries (fallback)
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.WARNING)

# Suppress noisy Azure Functions worker logs
azure_functions_worker_logger = logging.getLogger("azure_functions_worker")
azure_functions_worker_logger.setLevel(logging.WARNING)

# Ensure propagation is enabled for Azure Functions
logger.propagate = True
azure_search_logger.propagate = True
azure_identity_logger.propagate = True
azure_logger.propagate = True


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
        reranker_threshold: float = 2.0,
    ):
        self.agent_name = agent_name
        self.index_name = index_name
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        self.azure_search_endpoint = (
            azure_search_endpoint
            or f"https://{os.getenv('AZURE_SEARCH_SERVICE')}.search.windows.net"
        )
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

    def __init__(self, config: AgenticSearchConfig, organization_id: str):
        """
        Initialize the AgenticSearchManager with configuration.

        Args:
            config: AgenticSearchConfig instance with all necessary parameters
            organization_id: The organization ID to filter documents by
        """
        self.config = config
        self._agent_client: Optional[KnowledgeAgentRetrievalClient] = None
        self._search_client: Optional[SearchClient] = None
        self._agent_initialized = False
        self.organization_id = organization_id
        self._enable_http_logging = False

    def enable_http_logging(self, enable: bool = True):
        """
        Enable HTTP logging for Azure SDK operations.

        Args:
            enable (bool): Whether to enable HTTP logging
        """
        self._enable_http_logging = enable
        if enable:
            # Set Azure SDK logging to DEBUG level to see HTTP requests
            azure_search_logger.setLevel(logging.DEBUG)
            logger.info(
                "[AgenticSearchManager] HTTP logging enabled for Azure SDK operations"
            )
        else:
            azure_search_logger.setLevel(logging.INFO)
            logger.info(
                "[AgenticSearchManager] HTTP logging disabled for Azure SDK operations"
            )

    def agent_exists(self) -> bool:
        """
        Check if the knowledge agent exists without creating it.

        Returns:
            bool: True if agent exists, False otherwise
        """
        try:
            index_client = SearchIndexClient(
                endpoint=self.config.azure_search_endpoint,
                credential=self.config.credential,
            )

            # Try to get the agent
            agent = index_client.get_agent(self.config.agent_name)
            return agent is not None

        except Exception as e:
            # Agent doesn't exist or other error
            logger.error(f"[AgenticSearchManager] Agent check failed: {str(e)}")
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
                    default_reranker_threshold=self.config.reranker_threshold,
                )
            ],
            request_limits=KnowledgeAgentRequestLimits(max_output_size=10000),
        )

        # Create index client with proper logging configuration
        index_client = SearchIndexClient(
            endpoint=self.config.azure_search_endpoint,
            credential=self.config.credential,
            logging_enable=self._enable_http_logging,  # Enable HTTP logging if configured
        )

        try:
            index_client.create_or_update_agent(agent)
            logger.info(
                f"[AgenticSearchManager] Knowledge agent '{self.config.agent_name}' created or updated successfully"
            )
            self._agent_initialized = True
            return True
        except Exception as e:
            logger.error(
                f"[AgenticSearchManager] Error creating or updating knowledge agent: {str(e)}"
            )
            return False

    def delete_knowledge_agent(self) -> bool:
        """
        Delete the knowledge agent.
        """
        index_client = SearchIndexClient(
            endpoint=self.config.azure_search_endpoint,
            credential=self.config.credential,
            logging_enable=self._enable_http_logging,
        )
        try:
            index_client.delete_agent(self.config.agent_name)
            logger.info(
                f"[AgenticSearchManager] Knowledge agent '{self.config.agent_name}' deleted successfully"
            )
            return True
        except Exception as e:
            logger.error(
                f"[AgenticSearchManager] Error deleting knowledge agent: {str(e)}"
            )
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
                logging_enable=self._enable_http_logging,  # Enable HTTP logging if configured
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
                logging_enable=self._enable_http_logging,  # Enable HTTP logging if configured
            )
        return self._search_client

    def _build_activity_query_map(self, agentic_results) -> Dict[str, str]:
        """
        Build a mapping of activity IDs to their corresponding search queries.

        Args:
            agentic_results: The result object from agentic_retrieval containing activity data

        Returns:
            Dict[str, str]: Mapping of activity_id -> search_query
        """
        activity_query_map = {}

        if hasattr(agentic_results, "activity"):
            for activity in agentic_results.activity:
                try:
                    activity_dict = activity.as_dict()
                    if activity_dict.get("type") == "AzureSearchQuery":
                        activity_id = activity_dict.get("id")
                        search_query = activity_dict.get("query", {}).get(
                            "search", "N/A"
                        )
                        if activity_id:  # Only add if activity_id exists
                            activity_query_map[activity_id] = search_query
                except Exception as e:
                    # Log the error but continue processing other activities
                    logger.error(
                        f"[AgenticSearchManager] Error processing activity: {str(e)}"
                    )
                    continue

        return activity_query_map

    def retrieve_document_by_id(
        self, document_id: str, organization_id: str
    ) -> Optional[Dict]:
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
                "date_last_modified",
            ]

            # Perform the search with filter by ID
            search_results = search_client.search(
                search_text="*",
                select=select_fields,
                filter=f"id eq '{document_id}' and (organization_id eq '{organization_id}' or organization_id eq null)",  # TODO: we can strictly filter by org id later
                top=1,
            )

            # Convert search results to list and return first result
            results = list(search_results)

            if results:
                return dict(results[0])
            else:
                logger.info(
                    f"[AgenticSearchManager] No document found with ID: {document_id}"
                )
                return None

        except Exception as e:
            logger.error(
                f"[AgenticSearchManager] Error retrieving document by ID '{document_id}': {str(e)}"
            )
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
            if hasattr(agentic_results, "references"):
                references = [r.as_dict() for r in agentic_results.references]
            else:
                references = agentic_results

            for result in references:
                # Create a copy of the original result
                enriched_result = result.copy()

                # Extract the document ID from source_data
                document_id = None
                if "source_data" in result and "id" in result["source_data"]:
                    document_id = result["source_data"]["id"]

                if document_id:
                    # Retrieve additional document details
                    detailed_doc = self.retrieve_document_by_id(
                        document_id, self.organization_id
                    )

                    if detailed_doc:
                        # Add enrichment data as a new section
                        enriched_result["enriched_data"] = {
                            "source": detailed_doc.get("url"),
                            "date_last_modified": detailed_doc.get(
                                "date_last_modified"
                            ),
                            "organization_id": detailed_doc.get("organization_id"),
                            "content_length": len(detailed_doc.get("content", "")),
                        }
                        enriched_results.append(enriched_result)
                    else:
                        # Skip this document if it can't be retrieved due to organization ID mismatch
                        logger.info(
                            f"[AgenticSearchManager] Skipping document with ID '{document_id}' - not found in search index for the organization ID: {self.organization_id}"
                        )
                else:
                    # Skip documents without document ID
                    logger.info(
                        "[AgenticSearchManager] Skipping document - no document ID found for enrichment"
                    )

        except Exception as e:
            logger.error(
                f"[AgenticSearchManager] Error enriching agentic search results: {str(e)}"
            )
            # Return original results if enrichment fails
            return references if "references" in locals() else []

        return enriched_results

    def convert_to_documents(
        self, agentic_results, use_enriched_data: bool = True
    ) -> List[Document]:
        """
        Convert agentic search results to LangChain Document objects.

        Args:
            agentic_results: The result object from agentic_retrieval containing references
            use_enriched_data (bool): Whether to enrich results with additional metadata (default: True)

        Returns:
            List[Document]: List of de-duplicated LangChain Document objects with content and metadata
        """
        documents = []

        try:
            # Get enriched results if requested, otherwise use raw results
            if use_enriched_data:
                results = self.enrich_agentic_search_results(agentic_results)
            else:
                # Convert agentic results to dictionaries if not already
                if hasattr(agentic_results, "references"):
                    results = [r.as_dict() for r in agentic_results.references]
                else:
                    results = agentic_results

            # Create activity query mapping for additional context
            activity_query_map = self._build_activity_query_map(agentic_results)

            for result in results:
                # Extract core content for page_content
                source_data = result.get("source_data", {})
                content = source_data.get("content", "")
                doc_key = result.get("doc_key")
                # Build comprehensive metadata
                metadata = {}

                title = source_data.get("title")
                if title:
                    metadata["title"] = title

                if doc_key:
                    metadata["doc_key"] = doc_key

                # Add search context if available
                activity_source_id = result.get("activity_source")
                if activity_source_id and activity_source_id in activity_query_map:
                    metadata["search_query"] = activity_query_map[activity_source_id]
                elif activity_source_id:
                    metadata["activity_source"] = activity_source_id

                # Add enriched data if available and requested
                if use_enriched_data and "enriched_data" in result:
                    enriched = result["enriched_data"]
                    if "error" not in enriched:
                        # Add enriched metadata with prefixes to avoid conflicts
                        if enriched.get("source"):
                            metadata["source"] = enriched["source"]
                        if enriched.get("date_last_modified"):
                            metadata["date_last_modified"] = enriched[
                                "date_last_modified"
                            ]
                        if enriched.get("organization_id"):
                            metadata["organization_id"] = enriched["organization_id"]
                        if enriched.get("content_length"):
                            metadata["content_length"] = enriched["content_length"]
                    else:
                        metadata["enrichment_error"] = enriched["error"]

                # Create Document object
                document = Document(page_content=content, metadata=metadata)

                documents.append(document)

        except Exception as e:
            logger.error(
                f"[AgenticSearchManager] Error converting agentic search results to Document objects: {str(e)}"
            )
            return []

        # De-duplicate documents based on doc_key before returning
        deduplicated_documents = self.deduplicate_documents_by_doc_key(documents)

        return deduplicated_documents

    def deduplicate_documents_by_doc_key(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Remove duplicate Document objects based on their doc_key metadata.

        Args:
            documents (List[Document]): List of Document objects to de-duplicate

        Returns:
            List[Document]: De-duplicated list of Document objects
        """
        if not documents:
            return documents

        seen_doc_keys = set()
        deduplicated_docs = []
        duplicate_count = 0

        for doc in documents:
            doc_key = doc.metadata.get("doc_key")

            # If doc_key is None or empty, always include the document
            if not doc_key:
                deduplicated_docs.append(doc)
                continue

            # Check if we've seen this doc_key before
            if doc_key not in seen_doc_keys:
                seen_doc_keys.add(doc_key)
                deduplicated_docs.append(doc)
            else:
                duplicate_count += 1
                # Optionally log the duplicate (for debugging)
                title = doc.metadata.get("title", "Unknown")
                logger.info(
                    f"[AgenticSearchManager] Removing duplicate document: '{title}' with doc_key: {doc_key[:50]}..."
                )

        if duplicate_count > 0:
            logger.info(
                f"[AgenticSearchManager] De-duplication complete: Removed {duplicate_count} duplicate document(s). "
                f"Returning {len(deduplicated_docs)} unique documents."
            )

        return deduplicated_docs

    def agentic_retriever(
        self,
        conversation_history: List[Dict],
        reranker_threshold: float = None,
        max_docs_for_reranker: int = 150,
        auto_create_agent: bool = True,
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
        logger.info(
            f"[AgenticSearchManager] Starting agentic retrieval with {len(conversation_history)} messages"
        )
        logger.info(
            f"[AgenticSearchManager] Reranker threshold: {reranker_threshold or self.config.reranker_threshold}"
        )
        logger.info(
            f"[AgenticSearchManager] Max docs for reranker: {max_docs_for_reranker}"
        )

        # Use config default if not specified
        if reranker_threshold is None:
            reranker_threshold = self.config.reranker_threshold

        try:
            logger.info("[AgenticSearchManager] Getting agent client...")
            agent_client = self.get_agent_client()

            logger.info("[AgenticSearchManager] Preparing retrieval request...")

            # Convert to KnowledgeAgentMessage format
            agent_messages = agent_client.retrieve(
                retrieval_request=KnowledgeAgentRetrievalRequest(
                    messages=[
                        KnowledgeAgentMessage(
                            role=msg["role"],
                            content=[
                                KnowledgeAgentMessageTextContent(text=msg["content"])
                            ],
                        )
                        for msg in conversation_history
                    ],
                    target_index_params=[
                        KnowledgeAgentIndexParams(
                            index_name=self.config.index_name,
                            reranker_threshold=reranker_threshold,
                            max_docs_for_reranker=max_docs_for_reranker,
                            include_reference_source_data=True,
                        )
                    ],
                )
            )

            logger.info(
                "[AgenticSearchManager] Agentic retrieval completed successfully"
            )
            return agent_messages

        except Exception as e:
            error_message = str(e).lower()
            logger.error(
                f"[AgenticSearchManager] Agentic retrieval failed with error: {str(e)}"
            )

            # Check if the error is related to missing/invalid agent
            agent_related_errors = [
                "not found",
                "does not exist",
                "invalid agent",
                "agent not found",
                "unauthorized",
                "forbidden",
            ]

            is_agent_error = any(
                error_text in error_message for error_text in agent_related_errors
            )

            if auto_create_agent and is_agent_error:
                logger.info(
                    f"[AgenticSearchManager] Agent-related error detected: {str(e)}"
                )
                logger.info(
                    "[AgenticSearchManager] Attempting to create/update the knowledge agent..."
                )

                # Try to create or update the agent
                if self.create_or_update_knowledge_agent():
                    logger.info(
                        "[AgenticSearchManager] Agent created/updated successfully. Retrying retrieval..."
                    )

                    # Reset the agent client to use the new agent
                    self._agent_client = None

                    # Retry the retrieval
                    try:
                        logger.info(
                            "[AgenticSearchManager] Retrying agentic retrieval after agent creation..."
                        )
                        agent_client = self.get_agent_client()
                        agent_messages = agent_client.retrieve(
                            retrieval_request=KnowledgeAgentRetrievalRequest(
                                messages=[
                                    KnowledgeAgentMessage(
                                        role=msg["role"],
                                        content=[
                                            KnowledgeAgentMessageTextContent(
                                                text=msg["content"]
                                            )
                                        ],
                                    )
                                    for msg in conversation_history
                                ],
                                target_index_params=[
                                    KnowledgeAgentIndexParams(
                                        index_name=self.config.index_name,
                                        reranker_threshold=reranker_threshold,
                                        max_docs_for_reranker=max_docs_for_reranker,
                                        include_reference_source_data=True,
                                    )
                                ],
                            )
                        )
                        logger.info("[AgenticSearchManager] Retry successful!")
                        return agent_messages

                    except Exception as retry_error:
                        logger.error(
                            f"[AgenticSearchManager] Retry failed after agent creation: {str(retry_error)}"
                        )
                        raise retry_error
                else:
                    logger.error("[AgenticSearchManager] Failed to create/update agent")
                    raise e
            else:
                # Re-raise the original error if it's not agent-related or auto_create is disabled
                logger.error(
                    f"[AgenticSearchManager] Non-agent error or auto-creation disabled: {str(e)}"
                )
                raise e

    def enrich_and_display_results(self, agentic_results, max_results: int = 5) -> None:
        """
        Enrich agentic search results and display them in a formatted way.
        NOTE: This display raw agentic search results, unfiltered.

        Args:
            agentic_results: The result object from agentic_retrieval
            max_results (int): Maximum number of results to display (default: 5)
        """
        enriched_results = self.enrich_agentic_search_results(agentic_results)

        # Create a mapping of activity IDs to search queries
        activity_query_map = self._build_activity_query_map(agentic_results)

        total_results = len(enriched_results)
        showing_count = min(total_results, max_results)

        # Single header log
        header = f"""
{'='*80}
📋 AGENTIC SEARCH RESULTS SUMMARY
{'='*80}
📊 Total Results Found: {total_results}
🔍 Displaying: {showing_count} results
{'='*80}"""

        logger.info(header)

        for i, result in enumerate(enriched_results[:max_results], 1):
            # Extract data
            source_data = result.get("source_data", {})
            doc_id = source_data.get("id", "N/A")
            title = source_data.get("title", "N/A")

            # Truncate long document ID for readability
            display_doc_id = doc_id[:50] + "..." if len(doc_id) > 50 else doc_id

            # Get search query
            activity_source_id = result.get("activity_source", "N/A")
            search_query = "N/A"
            if activity_source_id != "N/A" and activity_source_id in activity_query_map:
                search_query = activity_query_map[activity_source_id]

            # Get enriched data
            enriched_info = ""
            if "enriched_data" in result:
                enriched = result["enriched_data"]
                if "error" not in enriched:
                    source_url = enriched.get("source", "N/A")
                    last_modified = enriched.get("date_last_modified", "N/A")
                    org_id = enriched.get("organization_id", "None")

                    # Extract path after ".net" for cleaner display
                    if source_url != "N/A" and ".net/" in source_url:
                        display_source = source_url.split(".net/", 1)[
                            1
                        ]  # Get everything after .net/
                    else:
                        display_source = source_url

                    enriched_info = f"""
📁 Source: {display_source}
📅 Last Modified: {last_modified}
🏢 Organization ID: {org_id}"""
                else:
                    enriched_info = f"❌ Enrichment Error: {enriched['error']}"

            # Get content preview
            content = source_data.get("content", "")
            content_preview = ""
            if content:
                preview_text = content[:150] + "..." if len(content) > 150 else content
                content_preview = f"📄 Content Preview: {preview_text}"

            # Create formatted result block
            result_block = f"""
┌─ Result #{i} ─────────────────────────────────────────────────
│ 📋 Title: {title}
│ 🆔 Document ID: {display_doc_id}
│ 🔍 Search Query: '{search_query}'{enriched_info}
│ {content_preview}
└─────────────────────────────────────────────────────────────────"""

            logger.info(result_block)

        # Footer
        footer = f"{'='*80}\n✅ Agentic Search Results Display Complete"
        logger.info(footer)

    def display_document_results(
        self, documents: List[Document], max_results: int = 1000  # a large number to display all documents
    ) -> None:
        """
        Display converted LangChain Document objects in a formatted way.
        NOTE: This displays final processed documents after conversion and deduplication.

        Args:
            documents: List of LangChain Document objects from convert_to_documents
            max_results (int): Maximum number of results to display (default: 5)
        """
        total_docs = len(documents)
        showing_count = min(total_docs, max_results)

        # Single header log
        header = f"""
{'='*80}
📚 CONVERTED LANGCHAIN DOCUMENTS SUMMARY
{'='*80}
📊 Total Documents: {total_docs}
🔍 Displaying: {showing_count} documents
🔄 Status: Post-conversion & deduplication
{'='*80}"""

        logger.info(header)

        for i, doc in enumerate(documents[:max_results], 1):
            # Extract metadata
            metadata = doc.metadata
            title = metadata.get("title", "N/A")
            doc_key = metadata.get("doc_key", "N/A")
            search_query = metadata.get("search_query", "N/A")
            source = metadata.get("source", "N/A")
            date_modified = metadata.get("date_last_modified", "N/A")
            org_id = metadata.get("organization_id", "None")
            content_length = metadata.get("content_length", "N/A")

            # Truncate long doc_key for readability
            display_doc_key = doc_key[:50] + "..." if len(doc_key) > 50 else doc_key

            # Extract path after ".net/" for cleaner source display
            if source != "N/A" and ".net/" in source:
                display_source = source.split(".net/", 1)[1]
            else:
                display_source = source

            # Get content preview
            content = doc.page_content
            content_preview = ""
            if content:
                preview_text = content[:200] + "..." if len(content) > 200 else content
                content_preview = f"📄 Content: {preview_text}"
            else:
                content_preview = "📄 Content: [No content available]"

            # Build metadata info
            metadata_info = f"""
🔍 Search Query: '{search_query}'
📋 Title: {title}
🔑 Doc Key: {display_doc_key}
📁 Source: {display_source}
📅 Last Modified: {date_modified}
🏢 Organization ID: {org_id}
📏 Content Length: {content_length}"""

            # Create formatted document block
            doc_block = f"""
┌─ Document #{i} ─────────────────────────────────────────────{metadata_info}
│ {content_preview}
└─────────────────────────────────────────────────────────────────"""

            logger.info(doc_block)

        # Footer
        footer = "✅ Document Results Display Complete"
        logger.info(footer)


def create_default_agentic_search_manager(organization_id: str) -> AgenticSearchManager:
    """
    Create an AgenticSearchManager with default configuration.
    Automatically checks if the knowledge agent exists and creates it if needed.

    Args:
        organization_id: The organization ID to filter documents by

    Returns:
        AgenticSearchManager: Configured manager instance with agent ready to use
    """
    config = AgenticSearchConfig()
    manager = AgenticSearchManager(config, organization_id)

    # Test logging to verify configuration
    logger.info(
        " [Agentic Search Initialization] Creating default agentic search manager..."
    )
    logger.info(f"[Agentic Search Initialization] Organization ID: {organization_id}")
    logger.info(f"[Agentic Search Initialization] Agent name: {config.agent_name}")
    logger.info(f"[Agentic Search Initialization] Index name: {config.index_name}")

    # Enable HTTP logging only if DEBUG level is enabled (following Azure SDK best practices)
    if azure_search_logger.isEnabledFor(logging.DEBUG):
        manager.enable_http_logging(True)
        logger.info(
            "[Agentic Search Initialization] HTTP logging enabled due to DEBUG log level"
        )

    # Check if agent exists and create it if needed
    try:
        logger.info(
            f"[Agentic Search Initialization] Checking if agent '{config.agent_name}' exists..."
        )
        if not manager.agent_exists():
            logger.info(
                f"[Agentic Search Initialization] Agent '{config.agent_name}' not found. Creating..."
            )
            success = manager.create_or_update_knowledge_agent()
            if success:
                logger.info(
                    f"[Agentic Search Initialization] Agent '{config.agent_name}' created successfully!"
                )
            else:
                logger.info(
                    f"[Agentic Search Initialization] Failed to create agent '{config.agent_name}'. Auto-creation during retrieval may be attempted."
                )
        else:
            logger.info(
                f"[Agentic Search Initialization] Agent '{config.agent_name}' already exists and is ready to use."
            )
    except Exception as e:
        logger.error(
            f"[Agentic Search Initialization] Error during agent check/creation: {str(e)}"
        )
        logger.info(
            "[Agentic Search Initialization] Will attempt auto-creation during retrieval if needed."
        )

    return manager


def retrieve_and_convert_to_document_format(
    conversation_history: List[Dict], organization_id: str
) -> List[Document]:
    """
    Retrieve documents and convert them to LangChain Document objects.
    """
    logger.info("[Agentic Search] Starting Agentic Search Manager...")
    manager = create_default_agentic_search_manager(organization_id)

    logger.info("[Agentic Search] Performing Agentic Retrieval...")
    raw_results = manager.agentic_retriever(conversation_history)

    logger.info(
        "[Agentic Search] Converting Agentic Retrieval Results to LangChain Document Format..."
    )
    documents = manager.convert_to_documents(raw_results, use_enriched_data=True)
    logger.info(f"[Agentic Search] Retrieved {len(documents)} documents")

    logger.info("[Agentic Search] Displaying Converted Document Results...")
    manager.display_document_results(documents, max_results=5)

    return documents


# Test/example code - only runs when script is executed directly
if __name__ == "__main__":
    logger.info("Testing Agentic Search functionality")
    logger.info("=" * 50)

    # Test conversation
    simple_messages = [
        {"role": "user", "content": "What is marketing?"},
        {"role": "assistant", "content": "Marketing is..."},
    ]
    # add the latest message to the conversation history
    latest_message = {"role": "user", "content": "What is latest news on marketing?"}
    simple_messages.append(latest_message)

    try:
        # Perform agentic retrieval (will auto-create agent if needed)
        logger.info("\nPerforming agentic retrieval...")
        agent_messages_simple = retrieve_and_convert_to_document_format(
            simple_messages, organization_id="123"
        )

        logger.info(agent_messages_simple)

    except Exception as e:
        logger.error(f"Agentic search failed: {str(e)}")
        logger.info(
            "This might indicate a configuration issue or service unavailability."
        )

    # Clean up: delete the agent
    logger.info("\nCleaning up test agent...")
    try:
        config = AgenticSearchConfig()
        cleanup_manager = AgenticSearchManager(config, "123")
        cleanup_manager.delete_knowledge_agent()
        logger.info("Test agent deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to delete test agent: {str(e)}")
