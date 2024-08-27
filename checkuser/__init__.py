import logging
import azure.functions as func
import json
import os
from shared.util import get_set_user, get_users, get_user, delete_user

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    # todo rename function and endpoint to match rest api endpoint naming conventions
    logging.info("Python HTTP trigger function processed a request.")

    if req.method == "GET":
        user_id = req.params.get("id")
        if user_id:
            user = get_user(user_id)
            return func.HttpResponse(
                json.dumps(user), mimetype="application/json", status_code=200
            )
        users = get_users()
        return func.HttpResponse(
            json.dumps(users), mimetype="application/json", status_code=200
        )
    elif req.method == "POST":
        req_body = req.get_json()
        id = req_body.get("id")
        name = req_body.get("name")
        email = req_body.get("email")
        is_new_user = False

        user = get_set_user({"id": id, "name": name, "email": email, "role": "user"})

        is_new_user = user["is_new_user"]
        user_data = user["user_data"]

        logging.info(f"Is new user: {is_new_user} User data: {user_data}")

        return func.HttpResponse(
            json.dumps(user_data), mimetype="application/json", status_code=200
        )
    elif req.method == "DELETE":
        user_id = req.params.get("id")
        if user_id:
            user = delete_user(user_id)
            return func.HttpResponse(
                json.dumps(user), mimetype="application/json", status_code=200
            )
    else:
        return func.HttpResponse("Method not allowed", status_code=405)
