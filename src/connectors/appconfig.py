import os
import logging

from typing import Any
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from azure.identity.aio import (
    ChainedTokenCredential as AsyncChainedTokenCredential,
    ManagedIdentityCredential as AsyncManagedIdentityCredential,
    AzureCliCredential as AsyncAzureCliCredential,
)
from azure.core.exceptions import ClientAuthenticationError
from azure.appconfiguration.provider import (
    AzureAppConfigurationKeyVaultOptions,
    load,
    SettingSelector,
)

from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

class AppConfigClient:

    credential = None
    aiocredential = None

    def __init__(self):
        """
        Bulk-loads all keys labeled 'orchestrator' and 'gpt-rag' into an in-memory dict,
        giving precedence to 'orchestrator' where a key exists in both.
        """
        # Defaults
        self.disabled = False
        self.auth_failed = False
        self.client = {}

        # ==== Load all config parameters in one place ====
        self.tenant_id = os.environ.get('AZURE_TENANT_ID', "*")
        self.client_id = os.environ.get('AZURE_CLIENT_ID', "*")

        self.allow_env_vars = False
        if "allow_environment_variables" in os.environ:
            # Parse string -> bool safely ("false" -> False)
            self.allow_env_vars = str(os.environ["allow_environment_variables"]).lower() in ("1", "true", "yes")

        endpoint = os.getenv("APP_CONFIG_ENDPOINT")

        # If there's no endpoint configured, skip remote config entirely.
        if not endpoint:
            logging.info("Azure App Configuration skipped: no APP_CONFIG_ENDPOINT set. Remote keys will not be fetched.")
            self.disabled = True
            return

        # Prepare credentials for endpoint-based access
        self.credential = ChainedTokenCredential(
            ManagedIdentityCredential(client_id=self.client_id),
            AzureCliCredential()
        )
        self.aiocredential = AsyncChainedTokenCredential(
            AsyncManagedIdentityCredential(client_id=self.client_id),
            AsyncAzureCliCredential()
        )

        orchestrator_label_selector = SettingSelector(label_filter='gpt-rag-orchestrator', key_filter='*')
        base_label_selector = SettingSelector(label_filter='gpt-rag', key_filter='*')
        no_label_selector = SettingSelector(label_filter=None, key_filter='*')

        # Try to load from Azure App Configuration. If auth fails, don't spam stack traces.
        try:
            self.client = load(
                selects=[orchestrator_label_selector, base_label_selector, no_label_selector],
                endpoint=endpoint,
                credential=self.credential,
                key_vault_options=AzureAppConfigurationKeyVaultOptions(credential=self.credential)
            )
            logging.info("Azure App Configuration loaded successfully.")
        except ClientAuthenticationError:
            # Not logged in / MSI unavailable -> disable remote config, log a friendly message.
            logging.warning(
                "Azure App Configuration disabled: not authenticated (run 'az login' or configure Managed Identity). "
                "Skipping remote configuration and not fetching any keys."
            )
            self.disabled = True
            self.auth_failed = True
            self.client = {}
        except Exception as e:
            # Any other error: disable and keep running without remote config.
            logging.warning(
                "Azure App Configuration unavailable (%s). Skipping remote configuration and not fetching any keys.",
                str(e)
            )
            self.disabled = True
            self.client = {}


    def get(self, key: str, default: Any = None, type: type = str) -> Any:
        return self.get_value(key, default=default, allow_none=False, type=type)
    
    def get_value(self, key: str, default: str = None, allow_none: bool = False, type: type = str) -> str:

        if key is None:
            raise Exception('The key parameter is required for get_value().')

        value = None

        allow_env_vars = False
        if "allow_environment_variables" in os.environ:
            allow_env_vars = str(os.environ["allow_environment_variables"]).lower() in ("1", "true", "yes")

        if allow_env_vars is True:
            value = os.environ.get(key)

        if value is None and not self.disabled:
            try:
                value = self.get_config_with_retry(name=key)
            except Exception:
                value = None

        if value is not None:
            if type is not None:
                if type is bool:
                    if isinstance(value, str):
                        value = value.lower() in ['true', '1', 'yes']
                else:
                    try:
                        value = type(value)
                    except ValueError as e:
                        raise Exception(f'Value for {key} could not be converted to {type.__name__}. Error: {e}')
            return value
        else:
            if default is not None or allow_none is True:
                return default
            
            raise Exception(f'The configuration variable {key} not found.')
        
    def retry_before_sleep(self, retry_state):
        # Log the outcome of each retry attempt.
        message = f"""Retrying {retry_state.fn}:
                        attempt {retry_state.attempt_number}
                        ended with: {retry_state.outcome}"""
        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            message += f"; Exception: {ex.__class__.__name__}: {ex}"
        if retry_state.attempt_number < 1:
            logging.info(message)
        else:
            logging.warning(message)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=5),
        stop=stop_after_attempt(5),
        before_sleep=retry_before_sleep
    )
    def get_config_with_retry(self, name):
        if self.disabled or not self.client:
            return None
        try:
            return self.client[name]
        except KeyError:
            return None
        except RetryError:
            return None

    # Helper functions for reading environment variables
    def read_env_variable(self, var_name, default=None):
        value = self.get_value(var_name, default)
        return value.strip() if value else default

    def read_env_list(self, var_name):
        value = self.get_value(var_name, "")
        return [item.strip() for item in value.split(",") if item.strip()]

    def read_env_boolean(self, var_name, default=False):
        value = self.get_value(var_name, str(default)).strip().lower()
        return value in ['true', '1', 'yes']