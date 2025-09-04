"""
Markdown to PDF conversion utility.

This module provides functionality to convert markdown content to PDF documents
using xhtml2pdf for PDF generation without native dependencies.
"""

import io
import logging
import markdown
import html
from typing import Dict, Any, Union
from xhtml2pdf import pisa

# Set up logging
logger = logging.getLogger(__name__)


def html_to_pdf_xhtml2pdf(html_content: str) -> bytes:
    """
    Convert HTML content to PDF bytes using xhtml2pdf.
    
    Args:
        html_content (str): The HTML content to convert
        
    Returns:
        bytes: PDF document as bytes
        
    Raises:
        Exception: If PDF generation fails
    """
    try:
        # Create a bytes buffer for the PDF output
        pdf_buffer = io.BytesIO()
        
        # Convert HTML to PDF using xhtml2pdf
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
        
        if pisa_status.err:
            raise Exception(f"xhtml2pdf conversion failed with {pisa_status.err} errors")
        
        # Get the PDF bytes
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Failed to convert HTML to PDF using xhtml2pdf: {str(e)}")
        raise Exception(f"HTML to PDF conversion failed: {str(e)}") from e


def markdown_to_html(markdown_content: str) -> str:
    """
    Convert markdown content to HTML with proper formatting.
    
    Args:
        markdown_content (str): The markdown content to convert
        
    Returns:
        str: HTML formatted content
        
    Raises:
        ValueError: If markdown_content is empty or None
    """
    if not markdown_content:
        raise ValueError("Markdown content cannot be empty or None")
    
    # Escape HTML first to prevent XSS (but allow markdown to work)
    # Note: We don't escape here since markdown needs to process special characters
    
    # Use markdown library with extensions for better formatting
    md = markdown.Markdown(extensions=[
        'fenced_code',
        'tables', 
        'toc',
        'nl2br',  # Convert newlines to <br>
        'sane_lists',
        'codehilite'  # Code highlighting
    ])
    
    html_content = md.convert(markdown_content)
    
    # Wrap in a complete HTML document with basic styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Generated Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            p {{
                margin-bottom: 15px;
            }}
            ul, ol {{
                margin-bottom: 15px;
                padding-left: 30px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                margin-bottom: 15px;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding-left: 20px;
                color: #666;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return full_html


def dict_to_pdf(input_dict: Dict[str, Any]) -> bytes:
    """
    Convert a dictionary containing markdown content to a PDF document.
    
    Expected dictionary format:
    {
        'content': 'markdown content here',
        'question.txt': 'optional question or description',  # ignored
        # other fields are ignored
    }
    
    Args:
        input_dict (Dict[str, Any]): Dictionary containing 'content' field with markdown
        
    Returns:
        bytes: PDF document as bytes
        
    Raises:
        ValueError: If input_dict is invalid or missing required fields
        KeyError: If 'content' field is missing
        Exception: If PDF generation fails
    """
    if not isinstance(input_dict, dict):
        raise ValueError("Input must be a dictionary")
    
    if 'content' not in input_dict:
        raise KeyError("Dictionary must contain a 'content' field")
    
    markdown_content = input_dict['content']
    
    if not isinstance(markdown_content, str):
        raise ValueError("Content field must be a string")
    
    if not markdown_content.strip():
        raise ValueError("Content field cannot be empty")
    
    try:
        logger.info(f"Converting markdown content to PDF (length: {len(markdown_content)} chars)")
        
        # Convert markdown to HTML
        html_content = markdown_to_html(markdown_content)
        
        # Convert HTML to PDF using xhtml2pdf
        pdf_bytes = html_to_pdf_xhtml2pdf(html_content)
        
        logger.info(f"Successfully generated PDF ({len(pdf_bytes)} bytes)")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Failed to convert markdown to PDF: {str(e)}")
        raise Exception(f"PDF generation failed: {str(e)}") from e


def save_dict_to_pdf_file(input_dict: Dict[str, Any], output_path: str) -> str:
    """
    Convert a dictionary containing markdown content to a PDF file.
    
    Args:
        input_dict (Dict[str, Any]): Dictionary containing 'content' field with markdown
        output_path (str): Path where the PDF file should be saved
        
    Returns:
        str: Path to the saved PDF file
        
    Raises:
        ValueError: If input_dict is invalid or missing required fields
        KeyError: If 'content' field is missing
        Exception: If PDF generation or file saving fails
    """
    try:
        # Generate PDF bytes
        pdf_bytes = dict_to_pdf(input_dict)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        
        logger.info(f"PDF saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save PDF to file: {str(e)}")
        raise Exception(f"Failed to save PDF file: {str(e)}") from e
