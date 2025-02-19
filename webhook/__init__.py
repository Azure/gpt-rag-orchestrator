import logging
import azure.functions as func
import json
import os
from azurefunctions.extensions.http.fastapi import Response
import stripe



from shared.util import handle_new_subscription_logs, handle_subscription_logs, update_organization_subscription, disable_organization_active_subscription, enable_organization_subscription, update_subscription_logs, updateExpirationDate

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    if req.method != "POST":
        return func.HttpResponse("Method not allowed", status_code=405)
    
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    endpoint_secret = os.getenv("STRIPE_SIGNING_SECRET")

    event = None
    
    requ = req.get_body()
    payload = await req.json()

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        print("  Webhook error while parsing basic request." + str(e))
        return json.dumps({"success": False}), 400
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = req.headers["stripe-signature"]
        
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except stripe.error.SignatureVerificationError as e:
            logging.info("  Webhook signature verification failed. " + str(e))
            return json.dumps({"success": False}), 400

    # Handle the event
    if event["type"] == "checkout.session.completed":
        logging.info("  Webhook received!", event["type"])
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
            logging.info(f"User {userId} updated with subscription {subscriptionId}")
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(
                content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                media_type="application/json",
                status_code=500,
            )
    elif event["type"] == "customer.subscription.updated":
        logging.info("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        status = event["data"]["object"]["status"]
        expirationDate = event["data"]["object"]["current_period_end"]
        logging.info(f"expirationDate: => {expirationDate}")
        logging.info(f"Subscription {subscriptionId} updated to status {status}")

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
                logging.info("Modification type not received, no audit action created.")
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

            # If modification_type does not match any of the above, do not take any action.
            logging.info(f"Unknown modification type: {modification_type}. No action taken.")
            return "No action", None, None, modified_by, modified_by_name, None
        
        action, previous_plan, current_plan, modified_by, modified_by_name, status_financial_assistant = determine_action(event)
        logging.info(f"Action determined: {action}")
        
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
        logging.info("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        event_type = event["type"].split(".")[-1] # Obtain "paused"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            disable_organization_active_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)

    elif event["type"] == "customer.subscription.resumed":
        logging.info("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1] # Obtain "resumed"
        try:
            handle_subscription_logs(subscriptionId, event_type)
            enable_organization_subscription(subscriptionId)
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return Response(content=json.dumps({"error": f"Error in webhook execution: {str(e)}"}), media_type="application/json", status_code=500)
        
    elif event["type"] == "customer.subscription.deleted":
        logging.info("  Webhook received!", event["type"])
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
        logging.info(f"Unexpected event type: {event['type']}")

    return Response(content=json.dumps({"success": True}), media_type="application/json")