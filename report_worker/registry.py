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
from datetime import datetime, timezone
import json
class ReportGeneratorBase(ABC):
    """
    Base class for all report generators.
    
    NOTE: The report_worker function now handles PDF conversion and blob storage automatically.
    Generators should return markdown content, and the worker will:
    1. Convert markdown to PDF using shared/markdown_to_pdf.py
    2. Store PDF in Azure Blob Storage at: documents/organization_files/{organization_id}/
    3. Add metadata: organization_id, report_id, timestamp
    """
    
    @abstractmethod
    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a report and return markdown content.
        
        Args:
            job_id: Unique identifier for the report job
            organization_id: Organization requesting the report
            parameters: Report-specific parameters
            
        Returns:
            str: Markdown formatted report content that will be converted to PDF
        """
        pass

class SampleReportGenerator(ReportGeneratorBase):
    """Sample report generator for demonstration purposes"""
    
    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """Generate a sample report in markdown format"""
        # Generate markdown content
        markdown_content = f"""# Sample Report

## Report Information
- **Job ID**: {job_id}
- **Organization ID**: {organization_id}
- **Generated At**: {datetime.now(timezone.utc).isoformat()}
- **Report Type**: Sample

## Parameters
```json
{json.dumps(parameters, indent=2)}
```

## Report Data

### Summary
This is a sample report generated to demonstrate the new markdown-based report generation system.

### Status
✅ **Completed Successfully**

### Key Metrics
| Metric | Value |
|--------|-------|
| Records Processed | 1 |
| Report Version | 1.0 |
| Status | Completed |

### Additional Information
The report generation system now automatically:
1. Converts markdown content to PDF
2. Stores PDFs in Azure Blob Storage
3. Organizes files by organization
4. Adds proper metadata

---
*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d at %H:%M:%S UTC')}*
"""
        return markdown_content

class ConversationAnalyticsGenerator(ReportGeneratorBase):
    """Generate analytics reports from conversation data"""
    
    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """Generate conversation analytics report"""
        # This would implement actual conversation analytics logic
        # For now, return a placeholder
        raise NotImplementedError("Conversation analytics generator not yet implemented")

class UsageReportGenerator(ReportGeneratorBase):
    """Generate usage reports for organization"""
    
    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
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
