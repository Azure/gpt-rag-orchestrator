"""
Report Registry

This module provides a registry system for different report generators.
Each report type has a unique key and associated generator function.
"""

import logging
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from reports.report_generator import run_analysis


class ReportGeneratorBase(ABC):
    """
    Base class for all report generators.

    NOTE: The report_worker function now handles PDF conversion and blob storage automatically.
    Generators should return markdown content, and the worker will:
    1. Convert markdown to PDF using shared/markdown_to_pdf.py
    2. Store PDF in Azure Blob Storage at: documents/organization_files/{organization_id}/
    3. Add metadata: organization_id, report_id, timestamp
    """

    # Define valid report types as class attribute
    VALID_REPORT_TYPES = {
        "sample",
        "brand_analysis",
        "competitor_analysis",
        "product_analysis"
    }

    def __init__(self, report_type: str):
        if report_type not in self.VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid report type '{report_type}'. "
                f"Valid types are: {', '.join(sorted(self.VALID_REPORT_TYPES))}"
            )
        self.report_type = report_type

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


class BrandAnalysisReportGenerator(ReportGeneratorBase):
    """Generate a brand analysis report"""

    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """Generate a brand analysis report"""
        # Extract brand_focus and industry_context from parameters
        brand_focus = parameters.get('brand_focus', "")
        industry_context = parameters.get('industry_context', "")
        
        if not brand_focus or not industry_context:
            raise ValueError("brand_focus and industry_context are required")

        # Generate dynamic query
        query = f"""
    Please generate the weekly Brand Analysis Report.

    Brand Focus: {brand_focus}

    Industry Context: {industry_context}
    """

        return run_analysis(query, self.report_type)


class CompetitorAnalysisReportGenerator(ReportGeneratorBase):
    """Generate a competitor analysis report"""

    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """Generate a competitor analysis report"""
        # Extract categories and other parameters
        categories = parameters.get('categories', [])
        industry_context = parameters.get('industry_context', "")
        
        if not categories or not industry_context:
            raise ValueError("categories and industry_context are required")

        # Extract competitor brands from categories
        brands = []
        for category in categories:
            if 'brands' in category:
                brands.extend(category['brands'])
            elif 'competitors' in category:
                brands.extend(category['competitors'])

        # Generate dynamic query
        brands_str = ", ".join(brands)
        query = f"Please provide a 2-page competitor analysis on these brands: {brands_str}. The industry is {industry_context}."

        return run_analysis(query, self.report_type)


class ProductAnalysisReportGenerator(ReportGeneratorBase):
    """Generate a product analysis report"""

    def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> str:
        """Generate a product analysis report"""
        # Extract categories and other parameters
        categories = parameters.get('categories', [])

        # Build product list from categories
        product_lines = []
        for category in categories:
            category_name = category.get('category', 'Unknown Category')
            products = category.get('product', [])

            for product in products:
                product_lines.append(f"{product} - {category_name}")

        # Generate product list string
        product_list_str = "\n    ".join(product_lines)

        # Generate dynamic query
        query = f"""
    Please generate a monthly product performance report for the following products:
    {product_list_str}
    """
        return run_analysis(query, self.report_type)


# Registry of available report generators
_REPORT_GENERATORS: Dict[str, Type[ReportGeneratorBase]] = {
    "sample": SampleReportGenerator("sample"),
    "brand_analysis": BrandAnalysisReportGenerator("brand_analysis"),
    "competitor_analysis": CompetitorAnalysisReportGenerator("competitor_analysis"),
    "product_analysis": ProductAnalysisReportGenerator("product_analysis"),
}


def get_generator(report_key: str) -> Optional[ReportGeneratorBase]:
    return _REPORT_GENERATORS.get(report_key)


def get_valid_report_types() -> set:
    """
    Get all valid report types from the base class definition.

    Returns:
        Set of valid report type strings
    """
    return ReportGeneratorBase.VALID_REPORT_TYPES.copy()


def list_available_generators() -> Dict[str, str]:
    """
    List all available report generators.

    Returns:
        Dict mapping report keys to generator class names
    """
    return {key: generator.__class__.__name__ for key, generator in _REPORT_GENERATORS.items()}
