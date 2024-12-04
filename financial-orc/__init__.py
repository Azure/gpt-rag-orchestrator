import logging
import azure.functions as func
import os
import json
from . import orchestrator
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)

class ReportType(BaseModel):
    """Categorize user query into one of the predefined report types.
    
    Attributes:
        report_type: The classified type of report based on user query
    """
    report_type: Literal[
        "monthly_economics",
        "weekly_economics", 
        "company_analysis",
        "ecommerce",
        "creative_brief"
    ] = Field(
        default="monthly_economics",
        description="Report classification based on query content",
        title="Report Type Classification"
    )

def initialize_llm() -> AzureChatOpenAI:
    """Initialize Azure OpenAI chat model with configuration.
    
    Returns:
        AzureChatOpenAI: Configured language model instance
    """
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="Agent",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.1,
    )

def categorize_query(query: str, llm: AzureChatOpenAI) -> str:
    """Categorize user query into predefined report types.
    
    Args:
        query: User's input query
        llm: Configured language model
        
    Returns:
        str: Classified report type
    """
    categorizer = llm.with_structured_output(ReportType)
    result = categorizer.invoke(query)
    return result.report_type

llm = initialize_llm()

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("[financial-orchestrator] Python HTTP trigger function processed a request.")

    # request body should look like this:
    # {
    #     "question": "string",
    #     "conversation_id": "string"
    #     "documentName": "string",
    #     "client_principal_id": "string",
    #     "client_principal_name": "string",
    # }

    req_body = req.get_json()
    conversation_id = req_body.get("conversation_id")
    question = req_body.get("question")

    client_principal_id = req_body.get("client_principal_id")
    client_principal_name = req_body.get("client_principal_name")

    # User is anonymous if no client_principal_id is provided
    if not client_principal_id or client_principal_id == "":
        client_principal_id = "00000000-0000-0000-0000-000000000000"
        client_principal_name = "anonymous"

    client_principal = {"id": client_principal_id, "name": client_principal_name}

    documentName = req_body.get("documentName", "")

    if not documentName or documentName == "":
        logging.error("[financial-orchestrator] no documentName found in json input")
        return func.HttpResponse(
            json.dumps({"error": "no documentName found in json input"}),
            mimetype="application/json",
            status_code=400,
        )

    # validate documentName exists in a hardcoded list
    # if not documentName in ['financial', 'feedback']:
    #     return func.HttpResponse('{"error": "invalid documentName"}', mimetype="application/json", status_code=200)

    if question:
        if documentName  == "defaultDocument" or documentName == "":
            documentName = categorize_query(question, llm)
        result = await orchestrator.run(
            conversation_id, question, documentName, client_principal
        )
        return func.HttpResponse(
            json.dumps(result), mimetype="application/json", status_code=200
        )
    else:
        logging.error("[financial-orchestrator] no question found in json input")
        return func.HttpResponse(
            json.dumps({"error": "no question found in json input"}),
            mimetype="application/json",
            status_code=400,
        )
