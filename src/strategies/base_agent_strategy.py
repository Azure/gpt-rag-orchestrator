import h11
import logging
import os
import re
import json

from typing import Dict, AsyncIterator
from abc import ABC, abstractmethod
from azure.identity.aio import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential
from azure.ai.projects.aio import AIProjectClient
from connectors.appconfig import AppConfigClient
from connectors.cosmosdb import CosmosDBClient
from dependencies import get_config
from pathlib import Path
from .agent_strategies import AgentStrategies

class BaseAgentStrategy(ABC):
    """
    Base strategy for agents.
    """
    strategy_type : AgentStrategies
    conversation : Dict
    user_context : Dict

    def __init__(self):
        """
        Initializes endpoint, model name, credentials, and default event handler.
        """
        # App configuration
        self.cfg : AppConfigClient = get_config()
        self.project_endpoint = self.cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT")
        self.account_endpoint = self.cfg.get("AI_FOUNDRY_ACCOUNT_ENDPOINT")
        self.model_name = self.cfg.get("CHAT_DEPLOYMENT_NAME")
        self.prompt_source = self.cfg.get("PROMPT_SOURCE", "file")
        self.openai_api_version = self.cfg.get("OPENAI_API_VERSION", "2025-04-01-preview")   

        logging.debug(f"[base_agent_strategy] Project endpoint: {self.project_endpoint}")
        logging.debug(f"[base_agent_strategy] Model name: {self.model_name}")

        if not self.project_endpoint or not self.model_name:
            raise EnvironmentError(
                "Both AI_FOUNDRY_PROJECT_ENDPOINT and CHAT_DEPLOYMENT_NAME must be set"
            )

        # Azure AD/Entra ID client ID (optional, for managed identity)
        client_id = os.environ.get("AZURE_CLIENT_ID")

        # Build chained token credential: CLI first, then managed identity
        self.credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(client_id=client_id)
        )

        # Initialize the async AIProjectClient
        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential
        )

        self.cosmos = CosmosDBClient()

        self.user_context = {}

    @classmethod
    async def create(cls):
        return cls()
    
    @abstractmethod
    async def initiate_agent_flow(self, user_message: str):
        """
        Kick off the agent execution pipeline and stream responses.

        This method will:
        1. Manage and update the conversation context.
        2. Dispatch the user's input through the agent pipeline.
        3. Yield response segments as they are generated.

        Args:
            user_message (str): The input message from the end user.

        Yields:
            AsyncIterator[str]: An async iterator over chunks of the agent’s response.
        """

    async def _read_prompt(self, prompt_name, placeholders=None):
        
        if self.prompt_source == 'file':
            return await self._read_prompt_file(prompt_name, placeholders)
        
        elif self.prompt_source == 'cosmos':
            return await self._read_prompt_cosmos(prompt_name, placeholders)

    async def _read_prompt_file(self, prompt_name, placeholders=None):
        """
        Load and process a prompt file, applying strategy-based variants and placeholder replacements.

        This method reads a prompt file associated with a given agent, supporting optional variants
        (e.g., audio-optimized or text-only) and dynamic placeholder substitution.

        **Prompt Directory Structure**:
        - Prompts are stored in the `prompts/` directory.
        - If a strategy type is defined (`self.strategy_type`), the file is expected in a subdirectory:
          `prompts/<strategy_type>/`.

        **Prompt File Naming Convention**:
        - The filename is based on the provided `prompt_name`: `<prompt_name>.txt`.
        - You can pre-define variants externally using names like `<prompt_name>_audio.txt` or 
          `<prompt_name>_text_only.txt`, but this method does not automatically append suffixes. 
          Suffix logic must be handled when building `prompt_name`.

        **Placeholder Substitution**:
        - If a `placeholders` dictionary is provided, placeholders in the format `{{key}}` are replaced by
          their corresponding values.
        - If any `{{key}}` remains after substitution, the method checks for a fallback file:
          `prompts/common/<key>.txt`. If found, its content replaces the placeholder.
        - If no replacement is available, a warning is logged.

        **Example**:
        For `prompt_name='agent1_audio'` and `self.strategy_type='customer_service'`, the file path would be:
        `prompts/customer_service/agent1_audio.txt`

        **Parameters**:
        - prompt_name (str): The base name of the prompt file (without path, but may include variant suffix).
        - placeholders (dict, optional): Mapping of placeholder names to their substitution values.

        **Returns**:
        - str: Final content of the prompt with placeholders replaced.

        **Raises**:
        - FileNotFoundError: If the specified prompt file does not exist.
        """
 
        # Construct the prompt file path
        prompt_file_path = os.path.join(self._prompt_dir(), f"{prompt_name}.txt")

        if not os.path.exists(prompt_file_path):
            logging.error(f"[base_agent_strategy] Prompt file '{prompt_name}' not found: {prompt_file_path}.")
            raise FileNotFoundError(f"Prompt file '{prompt_name}' not found.")

        logging.info(f"[base_agent_strategy] Using prompt file path: {prompt_file_path}")
        
        # Read and process the selected prompt file
        with open(prompt_file_path, "r") as f:
            prompt = f.read().strip()
            
            # Replace placeholders provided in the 'placeholders' dictionary
            if placeholders:
                for key, value in placeholders.items():
                    prompt = prompt.replace(f"{{{{{key}}}}}", value)
            
            # Find any remaining placeholders in the prompt
            pattern = r"\{\{([^}]+)\}\}"
            matches = re.findall(pattern, prompt)
            
            # Process each unmatched placeholder
            for placeholder_name in set(matches):
                # Skip if placeholder was already replaced
                if placeholders and placeholder_name in placeholders:
                    continue
                # Look for a corresponding file in 'prompts/common'
                common_file_path = os.path.join("prompts", "common", f"{placeholder_name}.txt")
                if os.path.exists(common_file_path):
                    with open(common_file_path, "r") as pf:
                        placeholder_content = pf.read().strip()
                        prompt = prompt.replace(f"{{{{{placeholder_name}}}}}", placeholder_content)
                else:
                    # Log a warning if the placeholder cannot be replaced
                    logging.warning(
                        f"[base_agent_strategy] Placeholder '{{{{{placeholder_name}}}}}' could not be replaced."
                    )
            return prompt

    async def _read_prompt_cosmos(self, prompt_name, placeholders=None):
 
        logging.info(f"[base_agent_strategy] Using cosmo prompt : {self.strategy_type.value} : {prompt_name}")

        prompt_json = await self.cosmos.get_document("prompts", f"{self.strategy_type.value}_{prompt_name}")
        prompt = prompt_json.get("content", "")
            
        # Replace placeholders provided in the 'placeholders' dictionary
        if placeholders:
            for key, value in placeholders.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", value)
        
        # Find any remaining placeholders in the prompt
        pattern = r"\{\{([^}]+)\}\}"
        matches = re.findall(pattern, prompt)
        
        # Process each unmatched placeholder
        for placeholder_name in set(matches):
            # Skip if placeholder was already replaced
            if placeholders and placeholder_name in placeholders:
                continue
            
            placeholder_content = await self.cosmos.get_document("prompts", f"placeholder_{placeholder_name}")
            prompt = prompt.replace(f"{{{{{placeholder_name}}}}}", placeholder_content)
            
        return prompt
    
    def _prompt_dir(self):
        # __file__ is .../src/strategies/base_agent_strategy.py
        # go up to the 'src' folder, then into 'prompts/<strategy>'
        src_folder = Path(__file__).resolve().parent.parent  # .../src
        if not hasattr(self, 'strategy_type'):
            raise ValueError("strategy_type is not defined")     
        prompt_folder = src_folder / "prompts" / self.strategy_type.value
        if not prompt_folder.exists():
            raise FileNotFoundError(f"Prompt directory not found: {prompt_folder}")
        return str(prompt_folder)
    
    def _get_model(self, model_name: str = 'CHAT_DEPLOYMENT_NAME') -> Dict:
        model_deployments = self.cfg.get("MODEL_DEPLOYMENTS", default='[]').replace("'", "\"")

        try:
            print(f"Model deployments: {model_deployments}")
            logging.info(f"Model deployments: {model_deployments}")

            json_model_deployments = json.loads(model_deployments)

            #get the canonical_name of 'CHAT_DEPLOYMENT_NAME'
            for deployment in json_model_deployments:
                if deployment.get("canonical_name") == model_name:
                    return deployment
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for model deployments: {e}")
            raise ValueError(f"Invalid model deployments configuration: {model_deployments}")
            
        return None
