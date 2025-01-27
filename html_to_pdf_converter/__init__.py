import logging
import azure.functions as func
import io
import os
import platform
import bleach

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)

def html_to_pdf(html_content: str) -> bytes:
    """Convert the html content to PDF bytes.
    
    Note: Requires WeasyPrint and its dependencies to be properly installed.
    For installation issues, see: https://github.com/assafelovic/gpt-researcher/issues/166
    """
    try:
        from weasyprint import HTML
    except ImportError as e:
        error_msg = "WeasyPrint import failed. Please ensure WeasyPrint and its dependencies are properly installed."
        if platform.system() == 'Windows':
            error_msg += "\nFor Windows installation guide, see: https://github.com/assafelovic/gpt-researcher/issues/166"
        raise ImportError(error_msg) from e

    # Create a bytes buffer instead of writing to file
    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    return pdf_buffer.getvalue()

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get request body
        req_body = req.get_json()
        html_content = req_body.get('html')
        
        if not html_content:
            return func.HttpResponse(
                "Please provide 'html' in the request body",
                status_code=400
            )

        # Add size validation
        if len(html_content) > 10 * 1024 * 1024:  # 10MB limit
            return func.HttpResponse(
                "HTML content too large. Maximum size is 10MB",
                status_code=400
            )

        # # Sanitize HTML content
        # sanitized_html_content = bleach.clean(html_content)

        # Basic HTML validation
        if not html_content.strip().startswith('<'):
            return func.HttpResponse(
                "Invalid HTML content",
                status_code=400
            )

        # Log request (sanitized)
        logging.info(f"Processing HTML content of length: {len(html_content)}")

        # Convert HTML to PDF bytes
        pdf_bytes = html_to_pdf(html_content)
        
        return func.HttpResponse(
            body=pdf_bytes,
            mimetype="application/pdf",
            status_code=200
        )
        
    except ValueError as ve:
        return func.HttpResponse(
            f"Invalid request body: {str(ve)}",
            status_code=400
        )
    except Exception as e:
        error_message = str(e)

        # windows error handling
        if platform.system() == 'Windows':
            error_message = f"""Error converting HTML to PDF: {str(e)}
            
            If you're experiencing WeasyPrint installation issues on Windows,
            please check the solution here: https://github.com/assafelovic/gpt-researcher/issues/166
            Common issues include GTK3 installation, missing dependencies, and path configuration."""

        else: 
            error_message = f"Error converting HTML to PDF: {str(e)}"

        logging.error(error_message)
        
        return func.HttpResponse(
            error_message,
            status_code=500
        )