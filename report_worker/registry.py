"""
Report Registry

This module provides a registry system for different report generators.
Each report type has a unique key and associated generator function.
"""

import logging
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod
from shared.blob_client_async import get_blob_service_client
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import ContentSettings

class ReportGeneratorBase(ABC):
    """Base class for all report generators"""
    
    @abstractmethod
    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report and return metadata about the generated artifact.
        
        Args:
            job_id: Unique identifier for the report job
            organization_id: Organization requesting the report
            parameters: Report-specific parameters
            
        Returns:
            Dict containing:
            - blob_url: URL to the generated report artifact
            - file_name: Name of the generated file
            - file_size: Size of the file in bytes
            - content_type: MIME type of the generated file
            - metadata: Any additional metadata about the report
        """
        pass

class SampleReportGenerator(ReportGeneratorBase):
    """Sample report generator for demonstration purposes"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a sample report"""
        import json
        from datetime import datetime, timezone
        from azure.storage.blob import BlobServiceClient
        import os
        
        # Create sample report content
        report_content = {
            "report_type": "sample",
            "job_id": job_id,
            "organization_id": organization_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters,
            "data": {
                "message": "This is a sample report",
                "status": "completed"
            }
        }
        
        # Convert to JSON
        json_content = json.dumps(report_content, indent=2)
        file_name = f"sample_report_{job_id}.json"
        
        bsc = await get_blob_service_client()
        container_client = bsc.get_container_client(container_name="reports")
        blob_name = f"{organization_id}/{file_name}"
        
        # Ensure container exists
        try:
            container_client.get_container_properties()
        except ResourceNotFoundError:
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass  # another worker created it
        except Exception as e:
            logging.error(f"Error getting container properties: {e}")
            raise e
            
        # Upload the report
        blob_client = container_client.get_blob_client(blob=blob_name)
        
        blob_client.upload_blob(
            json_content.encode('utf-8'), 
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json')
        )
        
        return {
            "blob_url": blob_client.url,
            "file_name": file_name,
            "file_size": len(json_content.encode('utf-8')),
            "content_type": "application/json",
            "metadata": {
                "records_processed": 1,
                "report_version": "1.0"
            }
        }

class ConversationAnalyticsGenerator(ReportGeneratorBase):
    """Generate analytics reports from conversation data"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversation analytics report"""
        # This would implement actual conversation analytics logic
        # For now, return a placeholder
        raise NotImplementedError("Conversation analytics generator not yet implemented")

class UsageReportGenerator(ReportGeneratorBase):
    """Generate usage reports for organization"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage report"""
        # This would implement actual usage reporting logic
        # For now, return a placeholder
        raise NotImplementedError("Usage report generator not yet implemented")

# Registry of available report generators
_REPORT_GENERATORS: Dict[str, Type[ReportGeneratorBase]] = {
    "sample": SampleReportGenerator,
    "conversation_analytics": ConversationAnalyticsGenerator,
    "usage_report": UsageReportGenerator,
}

def get_generator(report_key: str) -> Optional[ReportGeneratorBase]:
    cls = _REPORT_GENERATORS.get(report_key)
    return cls() if cls else None

def register_generator(report_key: str, generator: ReportGeneratorBase) -> None:
    """
    Register a new report generator.
    
    Args:
        report_key: Unique key for the report type
        generator: ReportGeneratorBase instance
    """
    if not isinstance(generator, ReportGeneratorBase):
        raise ValueError("Generator must inherit from ReportGeneratorBase")
        
    _REPORT_GENERATORS[report_key] = generator
    logging.info(f"Registered report generator: {report_key}")

def list_available_generators() -> Dict[str, str]:
    """
    List all available report generators.
    
    Returns:
        Dict mapping report keys to generator class names
    """
    return {key: generator.__class__.__name__ for key, generator in _REPORT_GENERATORS.items()}
