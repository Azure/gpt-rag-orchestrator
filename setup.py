import logging
import time
import requests
import argparse
import json
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logging.getLogger('azure').setLevel(logging.WARNING)


def call_search_api(search_service, search_api_version, resource_type, resource_name, method, credential, body=None):
    """
    Calls the Azure Search API with the specified parameters.

    Args:
        search_service (str): The name of the Azure Search service.
        search_api_version (str): The version of the Azure Search API to use.
        resource_type (str): The type of resource to access (e.g. "indexes", "docs").
        resource_name (str): The name of the resource to access.
        method (str): The HTTP method to use (either "get" or "put").
        credential (TokenCredential): An instance of a TokenCredential class that can provide an access token.
        body (dict, optional): The JSON payload to include in the request body (for "put" requests).

    Returns:
        None

    Raises:
        ValueError: If the specified HTTP method is not "get" or "put".
        HTTPError: If the response status code is 400 or greater.

    """    
    # get the token
    token = credential.get_token("https://search.azure.com/.default").token
    headers = {
        "Authorization": f"Bearer {token}",
        'Content-Type': 'application/json'
        # 'api-key': SEARCH_API_KEY
    }
    search_endpoint = f"https://{search_service}.search.windows.net/{resource_type}/{resource_name}?api-version={search_api_version}"
    response = None
    try:
        if method not in ["get", "put"]:
            logging.error(f"Invalid method {method} ")
        if method == "get":
            response = requests.get(search_endpoint, headers=headers)
        elif method == "put":
            response = requests.put(search_endpoint, headers=headers, json=body)
        if response is not None:
            status_code = response.status_code
            if status_code >= 400:
                logging.error(f"Error when calling search API {method} {resource_type} {resource_name}. Code: {status_code}. Reason: {response.reason}")
                response_text_dict = json.loads(response.text)
                logging.error(f"Error when calling search API {method} {resource_type} {resource_name}. Message: {response_text_dict['error']['message']}")                
            else:
                logging.info(f"Successfully called search API {method} {resource_type} {resource_name}. Code: {status_code}.")                
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error when calling search API {method} {resource_type} {resource_name}. Error: {error_message}")
    return response


def get_function_key(subscription_id, resource_group, function_app_name, credential):
    """
    Returns an API key for the given function.

    Parameters:
    subscription_id (str): The subscription ID.
    resource_group (str): The resource group name.
    function_app_name (str): The name of the function app.
    credential (str): The credential to use.

    Returns:
    str: A unique key for the function.
    """    
    logging.info(f"Obtaining function key after creating or updating its value.")
    accessToken = f"Bearer {credential.get_token('https://management.azure.com/.default').token}"
    # Get key
    requestUrl = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Web/sites/{function_app_name}/functions/orc/keys/mykey?api-version=2022-03-01"
    requestHeaders = {
        "Authorization": accessToken,
        "Content-Type": "application/json"
    }
    data = {
        'properties': {}
    }
    response = requests.put(requestUrl, headers=requestHeaders, data=json.dumps(data))
    response_json = json.loads(response.content.decode('utf-8'))
    try:
        function_key = response_json['properties']['value']
    except Exception as e:
        function_key = None
        logging.error(f"Error when getting function key. Details: {str(e)}.")        
    return function_key



def execute_setup(subscription_id, resource_group, function_app_name, key_vault_name, enable_managed_identities, enable_env_credentials):
    """
    This function performs the necessary steps to set up the orchestrator component.
    
    Args:
        subscription_id (str): The subscription ID of the Azure subscription to use.
        resource_group (str): The name of the resource group containing the solution resources.
        function_app_name (str): The name of the function app to use.
        key_vault_name (Str): The key vault name
        enable_managed_identities (bool): Whether to use managed identities to run the setup.
        enable_env_credentials (bool): Whether to use environment credentials to run the setup.

    Returns:
        None
    """    
    
    logging.info(f"Getting function app {function_app_name} properties.") 
    credential = DefaultAzureCredential(logging_enable=True, exclude_managed_identity_credential=not enable_managed_identities, exclude_environment_credential=not enable_env_credentials)
    function_endpoint = f"https://{function_app_name}.azurewebsites.net"

    logging.info(f"Function endpoint: {function_endpoint}")
    
    ###########################################################################
    # Get function key and store into key vault
    ########################################################################### 
    function_key = get_function_key(subscription_id, resource_group, function_app_name, credential)
    if function_key is None:
            logging.error(f"Could not get function key. Please make sure the function {function_app_name}/orc is deployed before running this script.")
            exit() 

    # Create a SecretClient object
    vault_url = f"https://{key_vault_name}.vault.azure.net"    
    secret_client = SecretClient(vault_url=vault_url, credential=credential)

    # Store the secret in the key vault
    secret_client.set_secret('orchestrator-host--functionKey', function_key)



def main(subscription_id=None, resource_group=None, function_app_name=None, key_vault_name=None, enable_managed_identities=False, enable_env_credentials=False):
    """
    Sets up a chunking function app in Azure.

    Args:
        subscription_id (str, optional): The subscription ID to use. If not provided, the user will be prompted to enter it.
        resource_group (str, optional): The resource group to use. If not provided, the user will be prompted to enter it.
        function_app_name (str, optional): The name of the function app. If not provided, the user will be prompted to enter it.
        enable_managed_identities (bool, optional): Whether to use managed identities to run the setup.
        enable_env_credentials (bool, optional): Whether to use environment credentials to run the setup.
    """   
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting setup.")

    if subscription_id is None:
        subscription_id = input("Enter subscription ID: ")
    if resource_group is None:
        resource_group = input("Enter resource group: ")
    if function_app_name is None:
        function_app_name = input("Enter function app name: ")
    if key_vault_name is None:
        key_vault_name = input("Enter key vault name: ")

    start_time = time.time()

    execute_setup(subscription_id, resource_group, function_app_name, key_vault_name, enable_managed_identities, enable_env_credentials)

    response_time = time.time() - start_time
    logging.info(f"Finished setup. {round(response_time,2)} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to do orchestrator component setup.')
    parser.add_argument('-s', '--subscription_id', help='Subscription ID')
    parser.add_argument('-r', '--resource_group', help='Resource group')
    parser.add_argument('-f', '--function_app_name', help='Function app name')
    parser.add_argument('-k', '--key_vault_name', help='Key vault name')    
    parser.add_argument('-i', '--enable_managed_identities', action='store_true', default=False, help='Enable managed identities')
    parser.add_argument('-e', '--enable_env_credentials', action='store_true', default=False, help='Enable environment credentials')    
    args = parser.parse_args()

    main(subscription_id=args.subscription_id, resource_group=args.resource_group, function_app_name=args.function_app_name, key_vault_name=args.key_vault_name, enable_managed_identities=args.enable_managed_identities, enable_env_credentials=args.enable_env_credentials)    