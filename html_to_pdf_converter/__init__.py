import logging
import azure.functions as func
from pathlib import Path
import json
from weasyprint import HTML

def html_to_pdf(html_content: str, output_path: str) -> Path:
    """Convert the html content to a pdf file."""
    HTML(string=html_content).write_pdf(output_path)
    return Path(output_path)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get request body
        req_body = req.get_json()
        html_content = req_body.get('html')
        output_path = req_body.get('output_path')
        
        if not html_content or not output_path:
            return func.HttpResponse(
                "Please provide both 'html' and 'output_path' in the request body",
                status_code=400
            )

        # Convert HTML to PDF using the original function
        result_path = html_to_pdf(html_content, output_path)
        
        return func.HttpResponse(
            body=json.dumps({
                'path': str(result_path)
            }),
            mimetype="application/json",
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