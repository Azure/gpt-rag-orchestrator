import os
import glob
import json
from typing import Any, Dict
from semantic_kernel.functions.kernel_function import TEMPLATE_FORMAT_MAP
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.functions.kernel_plugin import KernelPlugin


def import_plugin_from_prompt_directory(
    kernel: Any,
    parent_directory: str,
    plugin_directory_name: str,
    settings: Dict[str, Any],
    item_name_override: list[str] | None = None,
) -> KernelPlugin:
    """
    This function is a modified version of the import_plugin_from_prompt_directory function from the semantic_kernel package.
    it will allow us to modify the file from the rag plugin to be able to dynamically load the parameters
    like temperature, frequency_penalty, and presence_penalty from the settings file in the azure function.
    """

    plugin_directory = kernel._validate_plugin_directory(
        parent_directory=parent_directory, plugin_directory_name=plugin_directory_name
    )

    functions = []

    # Handle YAML files at the root
    yaml_files = glob.glob(os.path.join(plugin_directory, "*.yaml"))
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as file:
            yaml_content = file.read()
            functions.append(
                kernel.create_function_from_yaml(
                    yaml_content, plugin_name=plugin_directory_name
                )
            )

    # Handle directories containing skprompt.txt and config.json
    for item in os.listdir(plugin_directory):
        item_path = os.path.join(plugin_directory, item)
        if os.path.isdir(item_path):
            prompt_path = os.path.join(item_path, "skprompt.txt")
            config_path = os.path.join(item_path, "config.json")

            if os.path.exists(prompt_path) and os.path.exists(config_path):
                with open(config_path, "r") as config_file:
                    temp_config = config_file.read()
                    if item_name_override and item in item_name_override:
                        # Update the default settings with the settings from the orchestrator dynamically
                        temp_config = json.loads(temp_config)
                        temp_config["execution_settings"]["default"].update(settings)
                        temp_config = json.dumps(temp_config)
                    prompt_template_config = PromptTemplateConfig.from_json(temp_config)
                prompt_template_config.name = item

                with open(prompt_path, "r") as prompt_file:
                    prompt = prompt_file.read()
                    prompt_template_config.template = prompt

                prompt_template = TEMPLATE_FORMAT_MAP[
                    prompt_template_config.template_format
                ](prompt_template_config=prompt_template_config)

                functions.append(
                    kernel.create_function_from_prompt(
                        plugin_name=plugin_directory_name,
                        prompt_template=prompt_template,
                        prompt_template_config=prompt_template_config,
                        template_format=prompt_template_config.template_format,
                        function_name=item,
                        description=prompt_template_config.description,
                    )
                )

    return KernelPlugin(name=plugin_directory_name, functions=functions)
