"""
Azure Blob Storage Utilities

This module provides a simplified interface for Azure Blob Storage operations
using DefaultAzureCredential for authentication.

Key Features:
- Upload/download blob content with metadata
- Container management
- Consistent error handling
- Simplified credential management using DefaultAzureCredential

Dependencies:
- azure-storage-blob
- azure-identity
- azure-core
"""

import os
import logging
from typing import Optional, Dict, Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Initialize logger
_logger = logging.getLogger(__name__)


class BlobHandler:
    """
    Handles Azure Blob Storage operations with simplified credential management.

    This class provides methods for uploading, downloading, and managing blobs
    in Azure Storage using DefaultAzureCredential for authentication.
    """

    @classmethod
    def _get_credential(cls):
        """
        Get Azure credential using DefaultAzureCredential.

        Returns:
            Azure credential instance
        """
        return DefaultAzureCredential()

    @classmethod
    def _get_blob_service_client(cls, storage_account_url: str):
        """
        Create and return a BlobServiceClient instance.

        Args:
            storage_account_url: Azure Storage account URL

        Returns:
            BlobServiceClient instance
        """
        _credential = cls._get_credential()
        return BlobServiceClient(
            account_url=storage_account_url, credential=_credential
        )

    @classmethod
    def upload(
        cls,
        storage_account_url: str,
        container_name: str,
        blob_path: str,
        content: bytes,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ):
        """
        Upload content to Azure Blob Storage.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name
            blob_path: Path/name for the blob
            content: Content to upload (as bytes)
            metadata: Optional metadata dictionary
            overwrite: Whether to overwrite existing blobs

        Returns:
            BlobClient instance of the uploaded blob

        Raises:
            Exception: If upload fails
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )

            _blob_client = _container_client.upload_blob(
                name=blob_path,
                data=content,
                overwrite=overwrite,
                metadata=metadata,
            )

            _logger.debug(
                "Successfully uploaded blob: %s to container: %s",
                blob_path,
                container_name,
            )

            return _blob_client

        except Exception as e:
            _logger.error(
                "Failed to upload blob: %s to container: %s, error: %s",
                blob_path,
                container_name,
                str(e),
            )
            raise

    @classmethod
    def download(
        cls,
        storage_account_url: str,
        container_name: str,
        blob_path: str,
    ) -> bytes:
        """
        Download content from Azure Blob Storage.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name
            blob_path: Path/name of the blob to download

        Returns:
            Blob content as bytes

        Raises:
            ResourceNotFoundError: If blob doesn't exist
            Exception: If download fails
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )
            _blob_client = _container_client.get_blob_client(blob=blob_path)
            _blob_content = _blob_client.download_blob().readall()

            _logger.debug(
                "Successfully downloaded blob: %s from container: %s",
                blob_path,
                container_name,
            )

            return _blob_content

        except ResourceNotFoundError:
            _logger.debug(
                "Blob not found: %s in container: %s", blob_path, container_name
            )
            raise
        except Exception as e:
            _logger.error(
                "Failed to download blob: %s from container: %s, error: %s",
                blob_path,
                container_name,
                str(e),
            )
            raise

    @classmethod
    def ensure_container_exists(
        cls,
        storage_account_url: str,
        container_name: str,
    ):
        """
        Ensure that the specified container exists in Azure Blob Storage.
        Creates the container if it doesn't exist.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name to ensure exists

        Returns:
            ContainerClient instance

        Raises:
            Exception: If container creation fails
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )

            if not _container_client.exists():
                _container_client.create_container()
                _logger.info(
                    "Created new container: %s in storage account: %s",
                    container_name,
                    storage_account_url,
                )
            else:
                _logger.debug(
                    "Container already exists: %s in storage account: %s",
                    container_name,
                    storage_account_url,
                )

            return _container_client

        except Exception as e:
            _logger.error(
                "Failed to ensure container exists: %s in storage account: %s, error: %s",
                container_name,
                storage_account_url,
                str(e),
            )
            raise

    @classmethod
    def blob_exists(
        cls,
        storage_account_url: str,
        container_name: str,
        blob_path: str,
    ) -> bool:
        """
        Check if a blob exists in Azure Blob Storage.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name
            blob_path: Path/name of the blob to check

        Returns:
            True if blob exists, False otherwise
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )
            _blob_client = _container_client.get_blob_client(blob=blob_path)

            return _blob_client.exists()

        except Exception as e:
            _logger.error(
                "Failed to check blob existence: %s in container: %s, error: %s",
                blob_path,
                container_name,
                str(e),
            )
            return False

    @classmethod
    def get_blob_metadata(
        cls,
        storage_account_url: str,
        container_name: str,
        blob_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific blob.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name
            blob_path: Path/name of the blob

        Returns:
            Dictionary containing blob metadata, or None if blob doesn't exist
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )
            _blob_client = _container_client.get_blob_client(blob=blob_path)

            blob_properties = _blob_client.get_blob_properties()

            return {
                "metadata": blob_properties.metadata,
                "content_length": blob_properties.size,
                "content_type": blob_properties.content_settings.content_type,
                "last_modified": blob_properties.last_modified,
                "etag": blob_properties.etag,
            }

        except ResourceNotFoundError:
            _logger.debug(
                "Blob not found for metadata retrieval: %s in container: %s",
                blob_path,
                container_name,
            )
            return None
        except Exception as e:
            _logger.error(
                "Failed to get blob metadata: %s in container: %s, error: %s",
                blob_path,
                container_name,
                str(e),
            )
            return None

    @classmethod
    def delete_blob(
        cls,
        storage_account_url: str,
        container_name: str,
        blob_path: str,
    ) -> bool:
        """
        Delete a blob from Azure Blob Storage.

        Args:
            storage_account_url: Azure Storage account URL
            container_name: Container name
            blob_path: Path/name of the blob to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            _blob_service_client = cls._get_blob_service_client(storage_account_url)
            _container_client = _blob_service_client.get_container_client(
                container=container_name
            )
            _blob_client = _container_client.get_blob_client(blob=blob_path)

            _blob_client.delete_blob()

            _logger.info(
                "Successfully deleted blob: %s from container: %s",
                blob_path,
                container_name,
            )

            return True

        except ResourceNotFoundError:
            _logger.debug(
                "Blob not found for deletion: %s in container: %s",
                blob_path,
                container_name,
            )
            return False
        except Exception as e:
            _logger.error(
                "Failed to delete blob: %s from container: %s, error: %s",
                blob_path,
                container_name,
                str(e),
            )
            return False


# Convenience functions for common operations
def upload_text_to_blob(
    storage_account_url: str,
    container_name: str,
    blob_path: str,
    text_content: str,
    encoding: str = "utf-8",
    metadata: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
):
    """
    Upload text content to a blob.

    This is a convenience function that handles text encoding automatically.

    Args:
        storage_account_url: Azure Storage account URL
        container_name: Container name
        blob_path: Path/name for the blob
        text_content: Text content to upload
        encoding: Text encoding (default: utf-8)
        metadata: Optional metadata dictionary
        overwrite: Whether to overwrite existing blobs

    Returns:
        BlobClient instance of the uploaded blob
    """
    content_bytes = text_content.encode(encoding)
    return BlobHandler.upload(
        storage_account_url=storage_account_url,
        container_name=container_name,
        blob_path=blob_path,
        content=content_bytes,
        metadata=metadata,
        overwrite=overwrite,
    )


def download_text_from_blob(
    storage_account_url: str,
    container_name: str,
    blob_path: str,
    encoding: str = "utf-8",
) -> str:
    """
    Download text content from a blob.

    This is a convenience function that handles text decoding automatically.

    Args:
        storage_account_url: Azure Storage account URL
        container_name: Container name
        blob_path: Path/name of the blob to download
        encoding: Text encoding (default: utf-8)

    Returns:
        Text content as string
    """
    content_bytes = BlobHandler.download(
        storage_account_url=storage_account_url,
        container_name=container_name,
        blob_path=blob_path,
    )
    return content_bytes.decode(encoding)
