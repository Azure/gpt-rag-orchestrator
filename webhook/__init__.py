import logging
import azure.functions as func
import json
import os

import stripe


from shared.util import update_organization_subscription, disable_organization_active_subscription, enable_organization_subscription, updateExpirationDate

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    if req.method != "POST":
        return func.HttpResponse("Method not allowed", status_code=405)
    
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    endpoint_secret = os.getenv("STRIPE_SIGNING_SECRET")

    event = None
    payload = req.get_body()

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        print("⚠️  Webhook error while parsing basic request." + str(e))
        return json.dumps({"success": False}), 400
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = req.headers["stripe-signature"]
        
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except stripe.error.SignatureVerificationError as e:
            print("⚠️  Webhook signature verification failed. " + str(e))
            return json.dumps({"success": False}), 400

    # Handle the event
    if event["type"] == "checkout.session.completed":
        print("🔔  Webhook received!", event["type"])
        userId = event["data"]["object"]["client_reference_id"]
        organizationId = event["data"]["object"].get("metadata", {}).get("organizationId", "") or ""
        sessionId = event["data"]["object"]["id"]
        subscriptionId = event["data"]["object"]["subscription"]
        paymentStatus = event["data"]["object"]["payment_status"]
        custom_fields = event["data"]["object"].get("custom_fields", [])
        organizationName = custom_fields[0]["text"]["value"] if custom_fields else "No name"
        expirationDate = event["data"]["object"]["expires_at"]
        try:
            update_organization_subscription(userId, organizationId, subscriptionId, sessionId, paymentStatus, organizationName, expirationDate)
            logging.info(f"User {userId} updated with subscription {subscriptionId}")
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return func.HttpResponse(
                json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                mimetype="application/json",
                status_code=500,
            )
    elif event["type"] == "customer.subscription.updated":
        print("🔔  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        status = event["data"]["object"]["status"]
        expirationDate = event["data"]["object"]["current_period_end"]
        print(event)
        print(f"expirationDate: => {expirationDate}")
        print(f"Subscription {subscriptionId} updated to status {status}")
        enable_organization_subscription(subscriptionId)
        updateExpirationDate(subscriptionId, expirationDate)
    elif event["type"] == "customer.subscription.paused":
        print("🔔  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        disable_organization_active_subscription(subscriptionId)
    elif event["type"] == "customer.subscription.resumed":
        print("🔔  Webhook received!", event["type"])
        enable_organization_subscription(subscriptionId)
    elif event["type"] == "customer.subscription.deleted":
        print("🔔  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        disable_organization_active_subscription(subscriptionId)
    else:
        # Unexpected event type
        logging.info(f"Unexpected event type: {event['type']}")

    return func.HttpResponse(
        json.dumps({"success": True}), status_code=200
    )
