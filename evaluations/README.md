# Evaluation Process

This document provides an overview of the evaluation process for the GPT-RAG Orchestrator.

## Overview

The evaluation process leverages the Azure AI Project client library to provide quantitative, AI-assisted quality and safety metrics. These metrics are used to assess the performance of LLM models, GenAI applications, and agents. Metrics are defined as evaluators, which can be either built-in or custom, offering comprehensive evaluation insights.

For more details, refer to:

* [Azure AI Projects Evaluation Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/cloud-evaluation).
* [Azure AI Projects Evaluation Python SDK](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/README.md#evaluation).

## How It Works

The evaluation workflow is composed of several steps that prepare data, submit it to Azure AI Projects, and analyze results. First, a dataset of queries and ground truth is generated or provided. This dataset is uploaded to Azure AI Projects, creating a new version for the evaluation run. Next, a set of evaluators is configured, each targeting a specific quality or safety aspect of the responses. Finally, the evaluation is submitted to the service, where Azure processes the data and returns a structured report with scores and insights.

### Metrics Explained

* **Response Completeness**: Measures how thoroughly the generated response covers the expected information based on the ground truth. It evaluates whether critical aspects of the answer are present and accurate, ensuring the response does not omit important details. ([learn.microsoft.com](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.responsecompletenessevaluator?view=azure-python&utm_source=chatgpt.com))

* **Relevance**: Assesses how pertinent the response is to the original query. A high relevance score indicates that the answer directly addresses the user's question without diverging into unrelated topics. ([learn.microsoft.com](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-metrics-built-in))

* **Retrieval**: Evaluates the effectiveness of the retrieval step in a RAG (Retrieval-Augmented Generation) system. It measures whether the context provided to the model contains the most useful and accurate information for generating the response. ([learn.microsoft.com](https://learn.microsoft.com/en-sg/azure/ai-foundry/concepts/evaluation-metrics-built-in))

* **Content Safety**: Checks the response for potential harmful or inappropriate content, such as violent, self-harm, or offensive language. This metric helps identify safety risks before deploying to production, protecting users and maintaining trust. ([learn.microsoft.com](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-metrics-built-in))

Each evaluator typically requires specific input mappings (e.g., mapping "response" to the generated output, "ground\_truth" to expected answers, "context" to retrieved documents). Azure AI Projects handles the scoring logic within the service, returning numeric scores or categorical assessments depending on the evaluator.

## Creating the Test Dataset (Golden Dataset)

Before running evaluation, you need a "golden dataset": a JSONL file containing test entries with expected answers. Place it in the `dataset/` folder with the name `golden-dataset.jsonl`. Each line must be a JSON object with at least two fields:

* `query`: the question or input to test.
* `ground-truth`: the expected answer string.

Example format (one entry per line):

```json
{"query": "What is Contoso Electronics' mission?", "ground-truth": "To provide the highest quality electronic components for commercial and military aircraft while maintaining a commitment to safety and excellence."}
{"query": "How often are performance reviews conducted at Contoso Electronics?", "ground-truth": "Performance reviews are conducted annually."}
{"query": "Name three core values of Contoso Electronics.", "ground-truth": "Three core values are Quality, Integrity, and Innovation."}
```

Your `generate_eval_input.py` script reads this file (`dataset/golden-dataset.jsonl`) to build the evaluation input. Ensure the file is valid JSONL and encoded in UTF-8.

## Usage Instructions

These instructions guide you through running the evaluation for the GPT-RAG Orchestrator. The steps assume you have a working Python environment, Azure CLI, and necessary permissions to access Azure resources.

### Prerequisites

1. **Azure CLI Login**: You must be authenticated to Azure. Run:

   ```bash
   az login
   ```

   This ensures that managed identity or CLI credentials can be used by the evaluation scripts. ([learn.microsoft.com](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk))

2. **Set App Configuration Endpoint**: The environment variable `APP_CONFIG_ENDPOINT` must point to your Azure App Configuration resource. For example:

   ```bash
   export APP_CONFIG_ENDPOINT="https://<your-app-config-name>.azconfig.io"
   ```

   On Windows PowerShell:

   ```powershell
   $Env:APP_CONFIG_ENDPOINT = "https://<your-app-config-name>.azconfig.io"
   ```

3. **Azure Resources and Secrets**: Confirm the following keys in App Configuration Settings and ensure the corresponding secrets in Key Vault:

* `AI_FOUNDRY_PROJECT_ENDPOINT`: The endpoint for your Azure AI Projects service.
* `AI_FOUNDRY_ACCOUNT_ENDPOINT`: The endpoint for your Azure AI account.
* `CHAT_DEPLOYMENT_NAME`: The deployment name of your chat model.
* `KEY_VAULT_URI`: The URI of your Azure Key Vault.

In Azure Key Vault, create a secret named `evaluationsModelApiKey` with the value set to your model API key. If you prefer a different secret name, define it in the `EVALUATIONS_MODEL_API_KEY_SECRET_NAME` setting.

4. **Other Settings**: Ensure any additional App Configuration keys your application expects (e.g., `SEARCH_RAG_INDEX_NAME`, `SEARCH_SERVICE_QUERY_ENDPOINT`) are present.

### Running the Evaluation

Navigate to the root of the repository (where the `evaluations/` folder resides). Use one of the provided scripts based on your platform.

* **Linux/macOS (bash)**:

  ```bash
  cd path/to/gpt-rag-orchestrator
  ./evaluations/evaluate.sh
  ```

* **Windows (PowerShell)**:

  ```powershell
  cd path\to\gpt-rag-orchestrator
  .\evaluations\evaluate.ps1
  ```

These scripts perform the following:

1. Validate that `APP_CONFIG_ENDPOINT` is set. If missing, the script will exit with an error message.
2. Create and activate a Python virtual environment in `evaluations/.venv`.
3. Install required dependencies from `evaluations/requirements.txt`.
4. Ensure `PYTHONPATH` includes the project root and `src/`, so the evaluation code can import the application logic.
5. Run `generate_eval_input.py` to produce `dataset/eval-input.jsonl` based on your golden dataset and application endpoint.
6. Submit the evaluation using `evaluate.py`, which uploads the dataset, configures evaluators, and calls Azure AI Projects with necessary headers (`model-endpoint` and `api-key`).
7. Tear down the virtual environment after completion.

### Customization

* **Skipping Evaluation**: Use the `--skip-eval` or `-SkipEval` flag if you only want to generate the evaluation input without submitting to Azure.
* **Logging Level**: By default, logging is set to INFO. You can adjust logging settings in the scripts if you need DEBUG-level details.
* **Evaluator Configuration**: To modify which metrics run, edit the `evaluators` dictionary in `evaluations/evaluate.py`. Each entry maps a friendly name (e.g., "completeness") to an `EvaluatorConfiguration` with an `EvaluatorIds` enum value and data mappings.

### Troubleshooting

* **Authentication Errors**: If the script reports missing `model-endpoint` or `api-key`, verify that Key Vault contains the secret named by `EVALUATIONS_MODEL_API_KEY_SECRET_NAME` in App Configuration, and that AppConfigClient can access it.
* **App Configuration Access**: Ensure that the managed identity or Azure CLI login has access to read App Configuration settings.
* **Azure AI Projects Endpoint**: Confirm that `AI_FOUNDRY_PROJECT_ENDPOINT` and `AI_FOUNDRY_ACCOUNT_ENDPOINT` values are correct and correspond to your Azure AI resource.
* **Environment Variables**: Double-check that all required environment variables are set in your shell before running the scripts.

### Example Run

```bash
az login
export APP_CONFIG_ENDPOINT="https://my-config.azconfig.io"
cd ~/workspace/gpt-rag-orchestrator
./evaluations/evaluate.sh
```

After submission, the script will output a URL (if available) where you can view detailed results in Azure AI Studio.
