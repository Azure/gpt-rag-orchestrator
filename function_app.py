import azure.functions as func
import logging
import json
import os
import stripe
import traceback
from datetime import datetime, timezone

from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, Response

from shared.util import (
    get_user,
    handle_new_subscription_logs,
    handle_subscription_logs,
    update_organization_subscription,
    disable_organization_active_subscription,
    enable_organization_subscription,
    update_subscription_logs,
    updateExpirationDate,
    get_conversations,
    get_conversation,
    delete_conversation,
    trigger_indexer_with_retry,
)

from orc import new_orchestrator
from financial_orc import orchestrator as financial_orchestrator
from shared.conversation_export import export_conversation
from webscrapping.multipage_scrape import crawl_website
from report_worker.processor import extract_message_metadata, process_report_job
from shared.util import update_report_job_status
# MULTIPAGE SCRAPING CONSTANTS
DEFAULT_LIMIT = 30
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_BREADTH = 15

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.function_name(name="report_worker")
@app.queue_trigger(
    arg_name="msg", 
    queue_name="report-jobs",
    connection="AZURE_STORAGE_CONNECTION_STRING"
)
async def report_worker(msg: func.QueueMessage) -> None:
    """
    Azure Function triggered by messages in the report-jobs queue.
    
    Processes report generation jobs with proper error handling and retry logic.
    """
    logging.info('[report-worker] Python Service Bus Queue trigger function processed a request.')

    correlation_id = None
    job_id = None
    organization_id = None
    dequeue_count = 1
    
    try:
        # Extract message metadata and required fields
        job_id, organization_id, correlation_id, dequeue_count, message_id = extract_message_metadata(msg)
        
        # Return early if message parsing failed
        if not all([job_id, organization_id, correlation_id]):
            return
            
        # Process the report job
        await process_report_job(job_id, organization_id, correlation_id, dequeue_count)
            
    except Exception as e:
        logging.error(
            f"[ReportWorker] Unexpected error for job {job_id} "
            f"(correlation: {correlation_id}): {str(e)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        # Update job status if we have the info
        if job_id and organization_id:
            error_payload = {
                "error_type": "unexpected",
                "error_message": str(e), 
                "dequeue_count": dequeue_count,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            update_report_job_status(job_id, organization_id, 'FAILED', error_payload=error_payload)
        
        # Don't re-raise - let message go to poison queue

@app.route(route="orc", methods=[func.HttpMethod.POST])
async def stream_response(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client"""
    logging.info("[orc] Python HTTP trigger function processed a request.")

    req_body = await req.json()
    question = req_body.get("question")
    conversation_id = req_body.get("conversation_id")
    client_principal_id = req_body.get("client_principal_id")
    client_principal_name = req_body.get("client_principal_name")
    client_principal_organization = req_body.get("client_principal_organization")
    if not client_principal_id or client_principal_id == "":
        client_principal_id = "00000000-0000-0000-0000-000000000000"
        client_principal_name = "anonymous"
        client_principal_organization = "00000000-0000-0000-0000-000000000000"
    client_principal = {
        "id": client_principal_id,
        "name": client_principal_name,
        "organization": client_principal_organization,
    }

    organization_id = None
    user = get_user(client_principal_id)
    if "data" in user:
        organization_id = client_principal_organization

        logging.info(
            f"[FunctionApp] Retrieved organizationId: {organization_id} from user data"
        )

    # print configuration settings for the user
    settings = new_orchestrator.get_settings(client_principal)
    logging.info(f"[function_app] Configuration settings: {settings}")

    # validate settings
    temp_setting = settings.get("temperature")
    settings["temperature"] = float(temp_setting) if temp_setting is not None else 0.3
    settings["model"] = settings.get("model") or "DeepSeek-V3-0324"
    logging.info(f"[function_app] Validated settings: {settings}")
    if question:
        orchestrator = new_orchestrator.ConversationOrchestrator(
            organization_id=organization_id
        )
        try:
            logging.info(f"[FunctionApp] Processing conversation")
            return StreamingResponse(
                orchestrator.generate_response_with_progress(
                    conversation_id=conversation_id,
                    question=question,
                    user_info=client_principal,
                    user_settings=settings,
                ),
                media_type="text/event-stream",
            )
        except Exception as e:
            logging.error(f"[FunctionApp] Error in progress streaming: {str(e)}")
            return StreamingResponse(
                '{"error": "error in response generation"}',
                media_type="application/json",
            )
    else:
        return StreamingResponse(
            '{"error": "no question found in json input"}',
            media_type="application/json",
        )


@app.route(route="financial-orc", methods=[func.HttpMethod.POST])
async def financial_orc(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client
    input body should look like this:
    {
        "question": "string",
        "conversation_id": "string",
        "documentName": "string",
        "client_principal_id": "string",
        "client_principal_name": "string",
    }
    """
    logging.info("[financial-orc] Python HTTP trigger function processed a request.")

    req_body = await req.json()
    conversation_id = req_body.get("conversation_id")
    question = req_body.get("question")

    client_principal_id = req_body.get("client_principal_id")
    client_principal_name = req_body.get("client_principal_name")

    # User is anonymous if no client_principal_id is provided
    if not client_principal_id or client_principal_id == "":
        client_principal_id = "00000000-0000-0000-0000-000000000000"
        client_principal_name = "anonymous"

    client_principal = {"id": client_principal_id, "name": client_principal_name}

    # we did not rename this to document_id in order to avoid breaking changes, it is sent like this from the client
    documentName = req_body.get("documentName", "")

    if question:
        financial_orc = financial_orchestrator.FinancialOrchestrator()
        document_type = financial_orc.categorize_query(question)

        return StreamingResponse(
            financial_orc.generate_response(
                conversation_id=conversation_id,
                question=question,
                user_info=client_principal,
                document_id=documentName,
                document_type=document_type,
            ),
            media_type="text/event-stream",
        )
    else:
        logging.error("[financial-orchestrator] no question found in json input")
        return StreamingResponse(
            '{"error": "no question found in json input"}',
            media_type="application/json",
        )


@app.function_name(name="webhook")
@app.route(route="webhook", methods=[func.HttpMethod.POST, func.HttpMethod.GET])
async def webhook(req: Request) -> Response:
    logging.info("Python HTTP trigger function processed a request.")
    if req.method != "POST":
        return Response(
            content=json.dumps({"error": "Method not allowed"}),
            media_type="application/json",
            status_code=405,
        )

    stripe.api_key = os.getenv("STRIPE_API_KEY")
    endpoint_secret = os.getenv("STRIPE_SIGNING_SECRET")

    event = None
    payload = await req.body()

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        logging.error("  Webhook error while parsing basic request." + str(e))
        return Response(
            content=json.dumps({"success": False}),
            media_type="application/json",
            status_code=400,
        )
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = req.headers["stripe-signature"]

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except stripe.error.SignatureVerificationError as e:
            print("  Webhook signature verification failed. " + str(e))
            return Response(
                content=json.dumps({"success": False}),
                media_type="application/json",
                status_code=400,
            )

    # Handle the event
    if event["type"] == "checkout.session.completed":
        print("  Webhook received!", event["type"])
        userId = event["data"]["object"]["client_reference_id"]
        userName = event["data"]["object"].get("metadata", {}).get("userName", "") or ""
        organizationId = (
            event["data"]["object"].get("metadata", {}).get("organizationId", "") or ""
        )
        organizationName = (
            event["data"]["object"].get("metadata", {}).get("organizationName", "")
            or ""
        )
        sessionId = event["data"]["object"]["id"]
        subscriptionId = event["data"]["object"]["subscription"]
        paymentStatus = event["data"]["object"]["payment_status"]

        expirationDate = event["data"]["object"]["expires_at"]
        try:
            update_organization_subscription(
                userId,
                organizationId,
                subscriptionId,
                sessionId,
                paymentStatus,
                organizationName,
                expirationDate,
            )
            handle_new_subscription_logs(
                userId, organizationId, userName, organizationName
            )
            print(f"User {userId} updated with subscription {subscriptionId}")
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )
    elif event["type"] == "customer.subscription.updated":
        print("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        status = event["data"]["object"]["status"]
        expirationDate = event["data"]["object"]["current_period_end"]
        print(f"expirationDate: => {expirationDate}")
        print(f"Subscription {subscriptionId} updated to status {status}")

        def determine_action(event):
            data = event.get("data", {}).get("object", {})
            previous_data = event.get("data", {}).get("previous_attributes", {})
            metadata = data.get("metadata", {})

            modification_type = None
            modified_by = "Unknown"
            modified_by_name = "Unknown"

            if "modification_type" in metadata:
                modification_type = metadata.get("modification_type")
                modified_by = metadata.get("modified_by", "Unknown")
                modified_by_name = metadata.get("modified_by_name", "Unknown")

            # If modification_type is not received, log the message and do not create anything in the audit
            if modification_type is None:
                print("Modification type not received, no audit action created.")
                return "No action", None, None, modified_by, modified_by_name, None

            if modification_type == "add_financial_assistant":
                status_financial_assistant = "active"
                return (
                    "Financial Assistant Change",
                    None,
                    None,
                    modified_by,
                    modified_by_name,
                    status_financial_assistant,
                )
            elif modification_type == "remove_financial_assistant":
                status_financial_assistant = "inactive"
                return (
                    "Financial Assistant Change",
                    None,
                    None,
                    modified_by,
                    modified_by_name,
                    status_financial_assistant,
                )

            if modification_type == "subscription_tier_change":
                # Access plan info from subscription items in previous_data
                previous_plan = None
                if "items" in previous_data:
                    for item in previous_data["items"].get("data", []):
                        previous_plan = item.get("plan", {}).get("nickname", None)
                        if previous_plan:
                            break  # Exit once we find the plan

                current_plan = (
                    data.get("items", {})
                    .get("data", [{}])[0]
                    .get("plan", {})
                    .get("nickname", None)
                )

                return (
                    "Subscription Tier Change",
                    previous_plan,
                    current_plan,
                    modified_by,
                    modified_by_name,
                    None,
                )

            # Unknown action
            return "Unknown action", None, None, modified_by, modified_by_name, None

        (
            action,
            previous_plan,
            current_plan,
            modified_by,
            modified_by_name,
            status_financial_assistant,
        ) = determine_action(event)
        print(f"Action determined: {action}")

        try:
            enable_organization_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )

        if action != "No action":
            try:
                update_subscription_logs(
                    subscriptionId,
                    action,
                    previous_plan,
                    current_plan,
                    modified_by,
                    modified_by_name,
                    status_financial_assistant,
                )
                updateExpirationDate(subscriptionId, expirationDate)
            except Exception as e:
                logging.exception("[webbackend] exception in /api/webhook")
                return Response(
                    content=json.dumps(
                        {"error": f"Error in webhook execution: {str(e)}"}
                    ),
                    media_type="application/json",
                    status_code=500,
                )

    elif event["type"] == "customer.subscription.paused":
        print("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        event_type = event["type"].split(".")[-1]  # Obtain "paused"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            disable_organization_active_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )

    elif event["type"] == "customer.subscription.resumed":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1]  # Obtain "resumed"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            enable_organization_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )

    elif event["type"] == "customer.subscription.deleted":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1]  # Obtain "deleted"
        subscriptionId = event["data"]["object"]["id"]
        try:
            handle_subscription_logs(subscriptionId, event_type)
            disable_organization_active_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )
    else:
        # Unexpected event type
        print(f"Unexpected event type: {event['type']}")

    return Response(
        content=json.dumps({"success": True}), media_type="application/json"
    )

@app.blob_trigger(
    arg_name="myblob",
    path="documents/{name}",
    connection="AZURE_STORAGE_CONNECTION_STRING",
)
def blob_trigger(myblob: func.InputStream):
    """
    Azure Blob Storage trigger that processes uploaded documents and triggers search index updates.

    Args:
        myblob (func.InputStream): The uploaded blob file stream
    """
    try:
        # Extract file information
        blob_name = myblob.name
        file_extension = os.path.splitext(blob_name)[1].lower() if blob_name else ""

        logging.info(
            f"[blob_trigger] Processing blob: {blob_name}, Extension: {file_extension}"
        )

        # Define supported file types for indexing
        supported_extensions = {
            ".pdf",
            ".docx",
            ".doc",
            ".txt",
            ".md",
            ".html",
            ".pptx",
        }

        if file_extension not in supported_extensions:
            logging.info(
                f"[blob_trigger] File type {file_extension} not supported for indexing. Supported types: {supported_extensions}"
            )
            return

        # Get indexer name from environment or use default
        indexer_name = f'{os.getenv("AZURE_AI_SEARCH_INDEX_NAME")}-test-indexer'  # TODO: change to the actual indexer name once moved to prod

        logging.info(
            f"[blob_trigger] Triggering indexer '{indexer_name}' for supported document: {blob_name}"
        )

        # Trigger the indexer with retry logic for concurrent runs
        indexer_success = trigger_indexer_with_retry(indexer_name, blob_name)

        if indexer_success:
            logging.info(
                f"[blob_trigger] Successfully triggered indexer '{indexer_name}' for blob: {blob_name}"
            )
        else:
            logging.warning(
                f"[blob_trigger] Could not trigger indexer '{indexer_name}' for blob: {blob_name}. File will be indexed in next scheduled run."
            )

    except Exception as e:
        logging.error(
            f"[blob_trigger] Unexpected error processing blob {myblob.name if myblob else 'unknown'}: {str(e)}"
        )


@app.route(
    route="conversations",
    methods=[func.HttpMethod.GET, func.HttpMethod.POST, func.HttpMethod.DELETE],
)
async def conversations(req: Request) -> Response:
    logging.info("Python HTTP trigger function processed a request for conversations.")

    id = req.query_params.get("id")

    if req.method == "GET":
        try:
            user_id = req.query_params.get("user_id")
            if not user_id:
                return Response(
                    json.dumps({"error": "user_id query parameter is required"}),
                    media_type="application/json",
                    status_code=400,
                )

            if not id:
                conversations = get_conversations(user_id)
            else:
                conversations = get_conversation(id, user_id)
            return Response(
                json.dumps(conversations),
                media_type="application/json",
                status_code=200,
            )
        except Exception as e:
            logging.error(f"Error in GET /conversations: {str(e)}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                media_type="application/json",
                status_code=500,
            )

    elif req.method == "POST":
        try:
            req_body = await req.json()
            id_from_body = req_body.get("id")
            if not id_from_body:
                return Response("Missing conversation ID for export", status_code=400)

            user_id = req_body.get("user_id")
            export_format = req_body.get("format", "html")

            if not user_id:
                return Response("Missing user_id in request body", status_code=400)

            if export_format not in ["html", "json", "docx"]:
                return Response(
                    "Invalid export format. Supported formats: html, json, docx",
                    status_code=400,
                )

            result = export_conversation(id_from_body, user_id, export_format)

            if result["success"]:
                return Response(
                    json.dumps(result), media_type="application/json", status_code=200
                )
            else:
                return Response(
                    json.dumps({"error": result["error"]}),
                    media_type="application/json",
                    status_code=500,
                )

        except json.JSONDecodeError:
            return Response("Invalid JSON in request body", status_code=400)
        except Exception as e:
            logging.error(f"Error in conversation export: {str(e)}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                media_type="application/json",
                status_code=500,
            )
    # need to double check if this one is working
    elif req.method == "DELETE":
        try:
            req_body = await req.json()
            user_id = req_body.get("user_id")
            if not user_id:
                return Response("Missing user_id in request body", status_code=400)
            if id:
                try:
                    delete_conversation(id, user_id)
                    return Response(
                        "Conversation deleted successfully", status_code=200
                    )
                except Exception as e:
                    logging.error(f"Error deleting conversation: {str(e)}")
                    return Response("Error deleting conversation", status_code=500)
            else:
                return Response("Missing conversation ID", status_code=400)
        except json.JSONDecodeError:
            return Response("Invalid JSON in request body", status_code=400)
        except Exception as e:
            logging.error(f"Error in DELETE /conversations: {str(e)}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                media_type="application/json",
                status_code=500,
            )

    else:
        return Response("Method not allowed", status_code=405)


@app.route(route="scrape-page", methods=[func.HttpMethod.POST])
async def scrape_page(req: Request) -> Response:
    """
    Endpoint to scrape a single web page.

    Expected payload:
    {
        "url": "http://example.com",
        "client_principal_id": "user-id"
    }

    Returns:
        JSON response with scraping results and optional blob storage results
    """
    logging.info("[scrape-pages] Python HTTP trigger function processed a request.")

    try:

        req_body = await req.json()

        # Validate payload
        if not req_body or "url" not in req_body:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "Request body must contain 'url' field",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        url = req_body["url"]
        if not isinstance(url, str) or not url.strip():
            return Response(
                content=json.dumps(
                    {"status": "error", "message": "url must be a non-empty string"}
                ),
                media_type="application/json",
                status_code=400,
            )

        # Extract client principal ID and organization
        client_principal_id = req_body.get(
            "client_principal_id", "00000000-0000-0000-0000-000000000000"
        )

        organization_id = None
        try:
            user = get_user(client_principal_id)
            organization_id = user.get("data", {}).get("organizationId")
            if organization_id:
                logging.info(
                    f"[scrape-pages] Retrieved organizationId: {organization_id}"
                )
        except Exception as e:
            logging.info(f"[scrape-pages] No organization tracking - {str(e)}")

        from webscrapping import scrape_single_url
        from webscrapping.utils import generate_request_id

        request_id = req.headers.get("x-request-id") or generate_request_id()

        result_data = scrape_single_url(url.strip(), request_id, organization_id)

        result_status = result_data.get("status")
        if result_status == "completed":
            status_code = 200
        elif result_status == "failed":
            status_code = 422
        else:
            status_code = 500

        return Response(
            content=json.dumps(result_data),
            media_type="application/json",
            status_code=status_code,
        )

    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"status": "error", "message": "Invalid JSON format"}),
            media_type="application/json",
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Error in scrape-pages endpoint: {str(e)}")
        return Response(
            content=json.dumps(
                {"status": "error", "message": f"Internal server error: {str(e)}"}
            ),
            media_type="application/json",
            status_code=500,
        )


def create_preview_results(results: list, preview_length: int = 100) -> list:
    """
    Create a preview version of crawl results with truncated raw_content.

    Args:
        results: List of crawl results from Tavily
        preview_length: Number of characters to show in preview (default: 100)

    Returns:
        List of results with truncated raw_content for API response
    """
    if not results:
        return results

    preview_results = []
    for result in results:
        # Create a copy of the result
        preview_result = result.copy()

        # Truncate raw_content if it exists
        if "raw_content" in preview_result and preview_result["raw_content"]:
            content = preview_result["raw_content"]
            if len(content) > preview_length:
                preview_result["raw_content"] = content[:preview_length] + "..."

        preview_results.append(preview_result)

    return preview_results


@app.route(route="multipage-scrape", methods=[func.HttpMethod.POST])
async def multipage_scrape(req: Request) -> Response:
    """
    Endpoint to crawl a website using advanced multipage scraping with Tavily.

    Expected payload:
    {
        "url": "https://example.com",
        "limit": 30,           // optional, default 30
        "max_depth": 4,        // optional, default 4
        "max_breadth": 15,     // optional, default 15
        "client_principal_id": "user-id"  // optional
    }

    Returns:
        JSON response with crawling results including all discovered pages
    """
    logging.info("[multipage-scrape] Python HTTP trigger function processed a request.")

    try:
        req_body = await req.json()

        # Validate payload
        if not req_body or "url" not in req_body:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "Request body must contain 'url' field",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        url = req_body["url"]
        if not url or not isinstance(url, str):
            return Response(
                content=json.dumps(
                    {"status": "error", "message": "url must be a non-empty string"}
                ),
                media_type="application/json",
                status_code=400,
            )

        limit = req_body.get("limit", DEFAULT_LIMIT)
        max_depth = req_body.get("max_depth", DEFAULT_MAX_DEPTH)
        max_breadth = req_body.get("max_breadth", DEFAULT_MAX_BREADTH)

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "limit must be an integer between 1 and 100",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "max_depth must be an integer between 1 and 10",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        if not isinstance(max_breadth, int) or max_breadth < 1 or max_breadth > 50:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "max_breadth must be an integer between 1 and 50",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        # Extract client principal ID for logging/tracking
        client_principal_id = req_body.get(
            "client_principal_id", "00000000-0000-0000-0000-000000000000"
        )

        organization_id = None
        try:
            user = get_user(client_principal_id)
            organization_id = user.get("data", {}).get("organizationId")
            if organization_id:
                logging.info(
                    f"[multipage-scrape] Retrieved organizationId: {organization_id}"
                )
        except Exception as e:
            logging.info(f"[multipage-scrape] No organization tracking - {str(e)}")

        logging.info(
            f"[multipage-scrape] Starting crawl for URL: {url} with limit: {limit}, max_depth: {max_depth}, max_breadth: {max_breadth}"
        )

        # Extract request ID from headers if provided, or generate one
        from webscrapping.utils import generate_request_id

        request_id = req.headers.get("x-request-id") or generate_request_id()

        # Execute the multipage crawling
        crawl_result = crawl_website(url, limit, max_depth, max_breadth)

        # Check if crawling was successful
        if "error" in crawl_result:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": f"Crawling failed: {crawl_result['error']}",
                        "url": url,
                    }
                ),
                media_type="application/json",
                status_code=500,
            )

        # Initialize blob storage (always enabled)
        from webscrapping.blob_manager import create_crawler_manager_from_env
        from webscrapping.scraper import WebScraper

        crawler_manager = create_crawler_manager_from_env(request_id)
        blob_storage_result = None

        # Handle blob storage for all successful crawls
        if crawl_result.get("results"):
            # Format crawl results for blob storage
            crawl_parameters = {
                "limit": limit,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
            }

            formatted_pages = WebScraper.format_multipage_content_for_blob_storage(
                crawl_result=crawl_result,
                request_id=request_id,
                organization_id=organization_id,
                original_url=url,
                crawl_parameters=crawl_parameters,
            )

            if crawler_manager and formatted_pages:
                try:
                    # Upload to blob storage
                    blob_storage_result = (
                        crawler_manager.store_multipage_results_in_blob(
                            formatted_pages=formatted_pages, content_type="text/plain"
                        )
                    )

                    logging.info(
                        f"[multipage-scrape] Blob storage: {blob_storage_result['total_successful']} uploaded, "
                        f"{blob_storage_result['total_failed']} failed, {blob_storage_result['total_duplicates']} duplicates"
                    )

                except Exception as blob_error:
                    blob_storage_result = {
                        "status": "error",
                        "error": f"Blob storage upload failed: {str(blob_error)}",
                        "total_processed": len(formatted_pages),
                        "total_successful": 0,
                        "total_failed": len(formatted_pages),
                        "total_duplicates": 0,
                    }
                    logging.error(
                        f"[multipage-scrape] Blob storage failed for URL: {url}, error: {str(blob_error)}"
                    )
            elif not crawler_manager:
                # Storage not configured
                blob_storage_result = {
                    "status": "not_configured",
                    "message": "Blob storage not configured - missing Azure storage environment variables",
                    "total_processed": len(formatted_pages) if formatted_pages else 0,
                    "total_successful": 0,
                    "total_failed": 0,
                    "total_duplicates": 0,
                }
                logging.info(
                    f"[multipage-scrape] Blob storage not configured for URL: {url}"
                )
            else:
                # Failed to format pages
                blob_storage_result = {
                    "status": "error",
                    "error": "Failed to format pages for blob storage",
                    "total_processed": 0,
                    "total_successful": 0,
                    "total_failed": 0,
                    "total_duplicates": 0,
                }
        else:
            # No results to store
            blob_storage_result = {
                "status": "no_content",
                "message": "No pages found to store",
                "total_processed": 0,
                "total_successful": 0,
                "total_failed": 0,
                "total_duplicates": 0,
            }

        # Create preview results for API response (truncated raw_content)
        preview_results = create_preview_results(crawl_result.get("results", []))

        # Generate message based on blob storage result
        if blob_storage_result.get("total_successful", 0) > 0:
            if blob_storage_result.get("total_failed", 0) > 0:
                message = f"Scraped {blob_storage_result['total_successful']} pages successfully, {blob_storage_result['total_failed']} failed"
            else:
                message = f"Successfully scraped {blob_storage_result['total_successful']} pages and uploaded to blob storage"
        elif blob_storage_result.get("status") == "not_configured":
            message = f"Scraped {len(crawl_result.get('results', []))} pages (blob storage not configured)"
        else:
            message = f"Scraped {len(crawl_result.get('results', []))} pages but blob storage failed"

        # Format successful response with preview results
        response_data = {
            "status": "completed",
            "message": message,
            "url": url,
            "parameters": {
                "limit": limit,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
            },
            "results": preview_results,
            "pages_found": len(crawl_result.get("results", [])),
            "response_time": crawl_result.get("response_time", 0.0),
            "organization_id": organization_id,
            "request_id": request_id,
            "blob_storage_result": blob_storage_result,
        }

        logging.info(
            f"[multipage-scrape] Successfully crawled {response_data['pages_found']} pages from {url}"
        )

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            status_code=200,
        )

    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"status": "error", "message": "Invalid JSON format"}),
            media_type="application/json",
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Error in multipage-scrape endpoint: {str(e)}")
        return Response(
            content=json.dumps(
                {"status": "error", "message": f"Internal server error: {str(e)}"}
            ),
            media_type="application/json",
            status_code=500,
        )
