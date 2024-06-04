import logging
import azure.functions as func
import json
import os
from . import orchestrator

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    conversation_id = req_body.get('conversation_id')
    question = req_body.get('question')
    client_principal_id = req_body.get('client_principal_id')
    client_principal_name = req_body.get('client_principal_name') 
    if not client_principal_id or client_principal_id == '':
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'    
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }
    sql_search= req_body.get('sql_search')
    if not sql_search:
        sql_search = False
    teradata_search = req_body.get('teradata_search')
    if not teradata_search:
        teradata_search = False
    sql_server= req_body.get('sql_server')
    sql_database= req_body.get('sql_database')
    sql_table_info= req_body.get('sql_table_info')
    sql_username= req_body.get('sql_username')
    teradata_username= req_body.get('teradata_username')
    teradata_server= req_body.get('teradata_server')
    teradata_database= req_body.get('teradata_database')
    teradata_table_info= req_body.get('teradata_table_info')
    database_info= {
        'sql_search': sql_search,
        'teradata_search': teradata_search,
        'sql_server': sql_server,
        'sql_database': sql_database,
        'sql_table_info': sql_table_info,
        'sql_username': sql_username,
        'teradata_username': teradata_username,
        'teradata_server': teradata_server,
        'teradata_database': teradata_database,
        'teradata_table_info': teradata_table_info
    }

    if question:

        result = await orchestrator.run(conversation_id, question, client_principal,database_info)

        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse('{"error": "no question found in json input"}', mimetype="application/json", status_code=200)
