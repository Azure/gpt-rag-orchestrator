Part of [GPT‑RAG](https://aka.ms/gpt-rag)

<!-- 
page_type: sample
languages:
- azdeveloper
- powershell
- bicep
products:
- azure
- azure-ai-foundry
- azure-openai
- azure-ai-search
urlFragment: GPT-RAG
name: Multi-repo ChatGPT and Enterprise data with Azure OpenAI and AI Search
description: GPT-RAG core is a Retrieval-Augmented Generation pattern running in Azure, using Azure AI Search for retrieval and Azure OpenAI large language models to power ChatGPT-style and Q&A experiences.
-->
# GPT-RAG Orchestrator

Part of the [GPT-RAG](https://github.com/Azure/gpt-rag) solution.

The **GPT-RAG Orchestrator** service is an agentic orchestration layer built on Azure AI Foundry Agent Service and the Semantic Kernel framework. It enables agent-based RAG workflows by coordinating multiple specialized agents—each with a defined role—to collaboratively generate accurate, context-aware responses for complex user queries.

### How the Orchestrator Works

The orchestrator uses **Azure AI Foundry Agent Service** for single-agent and connected-agent flows, leveraging its managed runtime for agent lifecycle, state management, and tool orchestration. For multi-agent scenarios, it integrates the **Semantic Kernel Agent Framework** to compose and coordinate specialized agents working together on tasks. Custom agent strategies enable developers to add domain-specific logic without modifying the core orchestration code.

Developers can extend the orchestrator by creating a new subclass of `BaseAgentStrategy`, implementing the required `initiate_agent_flow` method (along with any additional helpers), and registering it in `AgentStrategyFactory.get_strategy` under a unique key.

## Documentation

For comprehensive information about GPT-RAG, including architecture details, configuration guides, best practices, troubleshooting resources, deployment guidance, customization options, and advanced usage scenarios, please refer to the [official project documentation](https://azure.github.io/GPT-RAG/).

## Prerequisites

Provision the infrastructure first by following the GPT-RAG repository instructions [GPT-RAG](https://github.com/azure/gpt-rag/tree/feature/vnext-architecture). This ensures all required Azure resources (e.g., Container App, Storage, AI Search) are in place before deploying the web application.

<details markdown="block">
<summary>Click to view <strong>software</strong> prerequisites</summary>
<br>
The machine used to customize and or deploy the service should have:

* Azure CLI: [Install Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
* Azure Developer CLI (optional, if using azd): [Install azd](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
* Git: [Download Git](https://git-scm.com/downloads)
* Python 3.12: [Download Python 3.12](https://www.python.org/downloads/release/python-3120/)
* Docker CLI: [Install Docker](https://docs.docker.com/get-docker/)
* VS Code (recommended): [Download VS Code](https://code.visualstudio.com/download)
</details>

## How to deploy the orchestrator service

Make sure you're logged in to Azure before anything else:

```bash
az login
```

Clone this repository.

### If you used `azd provision`

Just run:

```shell
azd env refresh
azd deploy 
```

> [!IMPORTANT]
> Make sure you use the **same** subscription, resource group, environment name, and location from `azd provision`.

### If you did **not** use `azd provision`

You need to set the App Configuration endpoint and run the deploy script.

#### Bash (Linux/macOS):

```bash
export APP_CONFIG_ENDPOINT="https://<your-app-config-name>.azconfig.io"
./scripts/deploy.sh
```

#### PowerShell (Windows):

```powershell
$env:APP_CONFIG_ENDPOINT = "https://<your-app-config-name>.azconfig.io"
.\scripts\deploy.ps1
```

## Found an Issue?

Encountered an error or bug? Help us improve the quality of this accelerator by reporting issues or suggesting enhancements on our **[GitHub Issues page](https://github.com/Azure/GPT-RAG/issues)**. Your feedback helps make GPT-RAG better for everyone!

## Previous Releases

> [!NOTE]  
> For earlier versions, use the corresponding release in the GitHub repository (e.g., v1.0.0 for the initial version).

## 🤝 Contributing

We appreciate contributions! See [CONTRIBUTING.md](https://github.com/Azure/GPT-RAG/blob/main/CONTRIBUTING.md) for guidelines on the Contributor License Agreement (CLA), code of conduct, and submitting pull requests.

## Trademarks

This project may contain trademarks or logos. Authorized use of Microsoft trademarks or logos must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Modified versions must not imply sponsorship or cause confusion. Third-party trademarks are subject to their own policies.
