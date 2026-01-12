import azure.functions as func
import azure.durable_functions as df
import logging
import json
import os
import stripe
import traceback
from datetime import datetime, timezone

from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, Response
from scheduler.batch_processor import load_and_process_jobs
from scheduler.create_batch_jobs import create_batch_jobs

from shared.util import (
    get_user,
    handle_new_subscription_logs,
    handle_subscription_logs,
    update_organization_subscription,
    disable_organization_active_subscription,
    enable_organization_subscription,
    update_subscription_logs,
    updateExpirationDate,
    trigger_indexer_with_retry,
)

from orc import ConversationOrchestrator, get_settings
from shared.conversation_export import export_conversation
from webscrapping.multipage_scrape import crawl_website
from report_worker.processor import extract_message_metadata, process_report_job
from shared.util import update_report_job_status

# MULTIPAGE SCRAPING CONSTANTS
DEFAULT_LIMIT = 30
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_BREADTH = 15

# Use DFApp for Durable Functions support
app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

# Must import AFTER app is created to register durable functions
import report_worker.activities  # GenerateReportActivity
import orchestrators.main_orchestrator 
import orchestrators.tenant_orchestrator  
import orchestrators.oneshot_orchestrator  
import entities.rate_limiter_entity  
ENABLE_LEGACY = os.getenv("ENABLE_LEGACY_QUEUE_WORKER") == "1"

if ENABLE_LEGACY:
    @app.function_name(name="report_worker")
    @app.queue_trigger(
        arg_name="msg",
        queue_name="report-jobs",
        connection="AZURE_STORAGE_CONNECTION_STRING",
    )
    async def report_worker(msg: func.QueueMessage) -> None:
        """
        Azure Function triggered by messages in the report-jobs queue.

        Processes report generation jobs with proper error handling and retry logic.
        """
        logging.info(
            "[report-worker] Python Service Bus Queue trigger function processed a request."
        )

        job_id = None
        organization_id = None
        dequeue_count = 1

        try:
            # Extract message metadata and required fields
            job_id, organization_id, dequeue_count, message_id = extract_message_metadata(
                msg
            )

            # Return early if message parsing failed
            if not all([job_id, organization_id]):
                return

            # Log processing start with dequeue count for monitoring
            logging.info(f"[ReportWorker] Starting job {job_id} for organization {organization_id} (attempt {dequeue_count})")

            # Check if this is a retry and log warning
            if dequeue_count > 1:
                logging.warning(f"[ReportWorker] Job {job_id} is being retried (attempt {dequeue_count})")

            # Process the report job
            await process_report_job(job_id, organization_id, dequeue_count)
            logging.info(f"[ReportWorker] Successfully completed job {job_id} for organization {organization_id}")

        except Exception as e:
            logging.error(
                f"[ReportWorker] Unexpected error for job {job_id} "
                f"(dequeue_count: {dequeue_count}): {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )

            # Update job status if we have the info
            if job_id and organization_id:
                error_payload = {
                    "error_type": "unexpected",
                    "error_message": str(e),
                    "dequeue_count": dequeue_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                update_report_job_status(
                    job_id, organization_id, "FAILED", error_payload=error_payload
                )

            # Don't re-raise - let message go to poison queue

@app.route(route="health", methods=[func.HttpMethod.GET])
async def health_check(req: Request) -> Response:
    """
    Health check endpoint for Azure App Service health monitoring.
    pinged by Azure's health check feature at 1-minute intervals

    Returns:
        200 OK when the application is healthy
    """
    return Response("OK", status_code=200, media_type="text/plain")

@app.route(route="start-orch", methods=[func.HttpMethod.POST])
@app.durable_client_input(client_name="client")
async def start_orch(req: Request, client: df.DurableOrchestrationClient):
    body = await req.json()
    orch = body.get("orchestrator", "OneShotOrchestrator")
    payload = body.get("input", {})
    instance_id = await client.start_new(orch, client_input=payload)
    return Response(content=json.dumps({"instanceId": instance_id}), media_type="application/json")

@app.timer_trigger(schedule="0 0 6 * * 0", arg_name="mytimer", run_on_startup=False)  # Every Sunday at 6:00 AM UTC
@app.durable_client_input(client_name="client")
async def batch_jobs_timer(mytimer: func.TimerRequest, client: df.DurableOrchestrationClient) -> None:
    """
    Timer trigger that runs every Sunday at 6:00 AM UTC.
    Cron expression: "0 0 6 * * 0" means:
    - 0 seconds
    - 0 minutes
    - 6 hours (6:00 AM)
    - * any day of month
    - * any month
    - 0 Sunday
    """
    logging.info("Batch jobs timer trigger started - Sunday 6:00 AM UTC")

    try:
        # Step 1: Create batch jobs
        batch_result = create_batch_jobs()
        logging.info(f"Created {batch_result.get('total_created', 0)} jobs")

        # Step 2: Load and process jobs
        result = await load_and_process_jobs(client)
        logging.info(f"Started orchestration: {result.get('instance_id')}")

        logging.info("Batch jobs timer completed successfully")
    except Exception as e:
        logging.error(f"Batch jobs timer failed: {str(e)}")
        raise

@app.route(route="orc", methods=[func.HttpMethod.POST])
async def stream_response(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client"""
    logging.info("[orc] Python HTTP trigger function processed a request.")

    req_body = await req.json()
    question = req_body.get("question")
    conversation_id = req_body.get("conversation_id")
    user_timezone = req_body.get("user_timezone")
    blob_names = req_body.get("blob_names", [])
    is_data_analyst_mode = req_body.get("is_data_analyst_mode", False)
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
    settings = get_settings(client_principal)
    logging.info(f"[function_app] Configuration settings: {settings}")

    # validate settings
    temp_setting = settings.get("temperature")
    settings["temperature"] = float(temp_setting) if temp_setting is not None else 0.3
    settings["model"] = settings.get("model") or "gpt-4.1"
    logging.info(f"[function_app] Validated settings: {settings}")
    if question:
        orchestrator = ConversationOrchestrator(
            organization_id=organization_id
        )
        try:
            logging.info("[FunctionApp] Processing conversation")
            return StreamingResponse(
                orchestrator.generate_response_with_progress(
                    conversation_id=conversation_id,
                    question=question,
                    user_info=client_principal,
                    user_settings=settings,
                    user_timezone=user_timezone,
                    blob_names=blob_names,
                    is_data_analyst_mode=is_data_analyst_mode,
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

@app.function_name(name="blob_event_grid_trigger")
@app.event_grid_trigger(arg_name="event")
def blob_event_grid_trigger(event: func.EventGridEvent):
    """
    Event Grid trigger that triggers the search indexer when blob events are received.
    Filtering is handled at the infrastructure level.
    """
    try:
        indexer_name = f'{os.getenv("AZURE_AI_SEARCH_INDEX_NAME")}-test-indexer'
        logging.info(f"[blob_event_grid] Event received, triggering indexer '{indexer_name}'")

        indexer_success = trigger_indexer_with_retry(indexer_name, event.subject)

        if indexer_success:
            logging.info(f"[blob_event_grid] Successfully triggered indexer '{indexer_name}'")
        else:
            logging.warning(f"[blob_event_grid] Could not trigger indexer '{indexer_name}'")

    except Exception as e:
        logging.error(f"[blob_event_grid] Error: {str(e)}, Event ID: {event.id}")


@app.route(
    route="conversations",
    methods=[func.HttpMethod.POST],
)
async def conversations(req: Request) -> Response:
    logging.info("Python HTTP trigger function processed a request for conversations.")

    if req.method == "POST":
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
