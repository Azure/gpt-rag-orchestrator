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

The **GPT-RAG Orchestrator** service is a modular orchestration system built on Azure AI Foundry Agent Service and the Semantic Kernel framework. It enables agent-based RAG workflows by coordinating multiple specialized agents—each with a defined role—to collaboratively generate accurate, context-aware responses for complex user queries.


### How the Orchestrator Works

_This section will be updated soon to include a detailed explanation of the orchestration flow, agent interactions, and underlying logic. For now, please refer to the source code and comments for insights into how the orchestrator coordinates multi-agent RAG workflows._

## Prerequisites

Before deploying the web application, you must provision the infrastructure as described in the [GPT-RAG](https://github.com/azure/gpt-rag/tree/feature/vnext-architecture) repo. This includes creating all necessary Azure resources required to support the application runtime.


## How to deploy the web app

```shell
azd env refresh
azd deploy 
````
> [!IMPORTANT]
> When running `azd env refresh`, make sure to use the **same subscription**, **resource group**, and **environment name** that you used during the infrastructure deployment. This ensures consistency across components.


## 🤝 Contributing

We appreciate contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on the Contributor License Agreement (CLA), code of conduct, and submitting pull requests.

## Trademarks

This project may contain trademarks or logos. Authorized use of Microsoft trademarks or logos must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Modified versions must not imply sponsorship or cause confusion. Third-party trademarks are subject to their own policies.