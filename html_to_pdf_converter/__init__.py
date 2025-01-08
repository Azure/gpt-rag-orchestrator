import logging
import azure.functions as func
import io
from weasyprint import HTML

def html_to_pdf(html_content: str) -> bytes:
    """Convert the html content to PDF bytes."""
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
        logging.error(f"Error converting HTML to PDF: {str(e)}")
        return func.HttpResponse(
            f"Error converting HTML to PDF: {str(e)}",
            status_code=500
        )