import azure.functions as func
import logging
import json
import os
import stripe
import platform

from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, Response

from scheduler import main as scheduler_main
from weekly_scheduler import main as weekly_scheduler_main
from monthly_scheduler import main as monthly_scheduler_main
from html_to_pdf_converter import html_to_pdf

from shared.util import (
    get_user,
    handle_new_subscription_logs, 
    handle_subscription_logs, 
    update_organization_subscription, 
    disable_organization_active_subscription, 
    enable_organization_subscription, 
    update_subscription_logs, 
    updateExpirationDate
)

from orc import new_orchestrator
from financial_orc import orchestrator as financial_orchestrator
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="orc", methods=[func.HttpMethod.POST])
async def stream_response(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client"""
    logging.info('[orc] Python HTTP trigger function processed a request.')

    req_body = await req.json()
    question = req_body.get('question')
    conversation_id = req_body.get('conversation_id')
    client_principal_id = req_body.get('client_principal_id')
    client_principal_name = req_body.get('client_principal_name') 
    if not client_principal_id or client_principal_id == '':
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'    
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }

    organization_id = None
    user = get_user(client_principal_id)
    if "data" in user and "organizationId" in user["data"]:
        organization_id = user["data"].get("organizationId")
        logging.info(f"Retrieved organizationId: {organization_id} from user data")
    
    if question:
        orchestrator = new_orchestrator.ConversationOrchestrator(
            organization_id=organization_id
        )
        try:
            resources =  orchestrator.process_conversation(
                conversation_id, question, client_principal
            )
        except Exception as e:
            return StreamingResponse('{"error": "error in orchestrator"}', media_type="application/json")
        try:
            return StreamingResponse(orchestrator.generate_response(resources["conversation_id"],resources["state"], resources["conversation_data"], client_principal, resources["memory_data"], resources["start_time"]), media_type="text/event-stream")
        except Exception as e:
            return StreamingResponse('{"error": "error in response generation"}', media_type="application/json")
    else:
        return StreamingResponse('{"error": "no question found in json input"}', media_type="application/json")

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

    documentName = req_body.get("documentName", "")

    if not documentName or documentName == "":
        logging.error("[financial-orchestrator] no documentName found in json input")
        return StreamingResponse('{"error": "no documentName found in json input"}', media_type="application/json")

    if question:
        financial_orc = financial_orchestrator.FinancialOrchestrator()
        if documentName  == "defaultDocument" or documentName == "":
            logging.info(f"[financial-orchestrator] categorizing query for {question}")
            documentName = financial_orc.categorize_query(question)
        return StreamingResponse(financial_orc.generate_response(conversation_id, question, client_principal, documentName), media_type="text/event-stream")
    else:
        logging.error("[financial-orchestrator] no question found in json input")
        return StreamingResponse('{"error": "no question found in json input"}', media_type="application/json")

@app.function_name(name="webhook")
@app.route(route="webhook", methods=[func.HttpMethod.POST, func.HttpMethod.GET])
async def webhook(req: Request) -> Response:
    logging.info("Python HTTP trigger function processed a request.")
    if req.method != "POST":
        return Response(content=json.dumps({"error": "Method not allowed"}), media_type="application/json", status_code=405)
    
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    endpoint_secret = os.getenv("STRIPE_SIGNING_SECRET")

    event = None
    payload = await req.body()

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        logging.error("  Webhook error while parsing basic request." + str(e))
        return Response(content=json.dumps({"success": False}), media_type="application/json", status_code=400)
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = req.headers["stripe-signature"]
        
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except stripe.error.SignatureVerificationError as e:
            print("  Webhook signature verification failed. " + str(e))
            return Response(content=json.dumps({"success": False}), media_type="application/json", status_code=400)

    # Handle the event
    if event["type"] == "checkout.session.completed":
        print("  Webhook received!", event["type"])
        userId = event["data"]["object"]["client_reference_id"]
        userName = event["data"]["object"].get("metadata", {}).get("userName", "") or ""
        organizationId = event["data"]["object"].get("metadata", {}).get("organizationId", "") or ""
        organizationName = event["data"]["object"].get("metadata", {}).get("organizationName", "") or ""
        sessionId = event["data"]["object"]["id"]
        subscriptionId = event["data"]["object"]["subscription"]
        paymentStatus = event["data"]["object"]["payment_status"]

        expirationDate = event["data"]["object"]["expires_at"]
        try:
            update_organization_subscription(userId, organizationId, subscriptionId, sessionId, paymentStatus, organizationName, expirationDate)
            handle_new_subscription_logs(userId, organizationId, userName, organizationName)
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
            metadata=data.get("metadata",{})

            modification_type= None
            modified_by="Unknown"
            modified_by_name="Unknown"

            if "modification_type" in metadata:
                modification_type = metadata.get("modification_type")
                modified_by = metadata.get("modified_by", "Unknown")
                modified_by_name = metadata.get("modified_by_name","Unknown")

            # If modification_type is not received, log the message and do not create anything in the audit
            if modification_type is None:
                print("Modification type not received, no audit action created.")
                return "No action", None, None, modified_by, modified_by_name, None    

            if modification_type == "add_financial_assistant":
                status_financial_assistant = "active"
                return "Financial Assistant Change", None, None, modified_by,modified_by_name, status_financial_assistant
            elif modification_type == "remove_financial_assistant":
                status_financial_assistant = "inactive"
                return "Financial Assistant Change", None, None, modified_by,modified_by_name, status_financial_assistant
            

            if modification_type == "subscription_tier_change":
                # Access plan info from subscription items in previous_data
                previous_plan = None
                if "items" in previous_data:
                    for item in previous_data["items"].get("data", []):
                        previous_plan = item.get("plan", {}).get("nickname", None)
                        if previous_plan:
                            break  # Exit once we find the plan
        
                current_plan = data.get("items", {}).get("data", [{}])[0].get("plan", {}).get("nickname", None)

                return "Subscription Tier Change", previous_plan, current_plan, modified_by, modified_by_name, None

            # Unknown action
            return "Unknown action", None, None, modified_by,modified_by_name, None
        
        action, previous_plan, current_plan, modified_by, modified_by_name, status_financial_assistant = determine_action(event)
        print(f"Action determined: {action}")
        
        try:
            enable_organization_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)

        if action != "No action":
            try:
                update_subscription_logs(subscriptionId, action, previous_plan, current_plan, modified_by, modified_by_name, status_financial_assistant)
                updateExpirationDate(subscriptionId, expirationDate)
            except Exception as e:
                logging.exception("[webbackend] exception in /api/webhook")
                return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)

    elif event["type"] == "customer.subscription.paused":
        print("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        event_type = event["type"].split(".")[-1] # Obtain "paused"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            disable_organization_active_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)

    elif event["type"] == "customer.subscription.resumed":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1] # Obtain "resumed"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            enable_organization_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)
        
    elif event["type"] == "customer.subscription.deleted":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1] # Obtain "deleted"
        subscriptionId = event["data"]["object"]["id"]
        try:
            handle_subscription_logs(subscriptionId, event_type)
            disable_organization_active_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)
    else:
        # Unexpected event type
        print(f"Unexpected event type: {event['type']}")

    return Response(content=json.dumps({"success": True}), media_type="application/json")

@app.function_name(name="scheduler")
@app.schedule(schedule="0 0 11,23 * * *", arg_name="timer")
async def scheduler(timer: func.TimerRequest) -> None:
    # Your scheduler implementation
    try:
        scheduler_main(timer)
    except Exception as e:
        logging.error(f"Error in scheduler: {e}")

@app.function_name(name="weekly_scheduler")
@app.schedule(schedule="0 0 13 * * 1", arg_name="timer")
async def weekly_scheduler(timer: func.TimerRequest) -> None:
    # Your weekly scheduler implementation
    try:
        weekly_scheduler_main(timer)
    except Exception as e:
        logging.error(f"Error in weekly scheduler: {e}")

@app.function_name(name="monthly_scheduler")
@app.schedule(schedule="0 0 13 1 * *", arg_name="timer")
async def monthly_scheduler(timer: func.TimerRequest) -> None:
    # Your monthly scheduler implementation
    try:
        monthly_scheduler_main(timer)
    except Exception as e:
        logging.error(f"Error in monthly scheduler: {e}")

@app.function_name(name="html_to_pdf_converter")
@app.route(route="html_to_pdf_converter", methods=[func.HttpMethod.POST])
async def html2pdf_conversion(req: Request) -> Response:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get request body
        req_body = await req.json()
        html_content = req_body.get('html')
        
        if not html_content:
            return Response(
                content="Please provide 'html' in the request body",
                status_code=400
            )

        # Add size validation
        if len(html_content) > 10 * 1024 * 1024:  # 10MB limit
            return Response(
                content="HTML content too large. Maximum size is 10MB",
                status_code=400
            )

        # Basic HTML validation
        if not html_content.strip().startswith('<'):
            return Response(
                content="Invalid HTML content",
                status_code=400
            )

        # Log request (sanitized)
        logging.info(f"Processing HTML content of length: {len(html_content)}")

        # Convert HTML to PDF bytes
        pdf_bytes = html_to_pdf(html_content)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            status_code=200
        )
        
    except ValueError as ve:
        return Response(
            content=f"Invalid request body: {str(ve)}",
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
        
        return Response(
            content=error_message,
            status_code=500
        )
