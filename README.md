# Enterprise RAG Orchestrator

This Orchestrator is part of the **Enterprise RAG (GPT-RAG)** Solution Accelerator.

To learn more about the Enterprise RAG, please go to [https://aka.ms/gpt-rag](https://aka.ms/gpt-rag).

### Cloud Deployment

To deploy the orchestrator in the cloud for the first time, please follow the deployment instructions provided in the [Enterprise RAG repo](https://github.com/Azure/GPT-RAG?tab=readme-ov-file#getting-started).  
   
These instructions include the necessary infrastructure templates to provision the solution in the cloud.  
   
Once the infrastructure is provisioned, you can redeploy just the orchestrator component using the instructions below:

First, please confirm that you have met the prerequisites:

 - Azure Developer CLI: [Download azd for Windows](https://azdrelease.azureedge.net/azd/standalone/release/1.5.0/azd-windows-amd64.msi), [Other OS's](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd).
 - Git: [Download Git](https://git-scm.com/downloads)
 - Python 3.10: [Download Python](https://www.python.org/downloads/release/python-31011/)

Then just clone this repository and reproduce the following commands within the gpt-rag-orchestrator directory:  

```
azd auth login  
azd env refresh  
azd deploy  
```

> Note: when running the ```azd env refresh```, use the same environment name, subscription, and region used in the initial provisioning of the infrastructure.

### Running Locally with VS Code  
   
[How can I test the solution locally in VS Code?](docs/LOCAL_DEPLOYMENT.md)

### Evaluating

[How to test the orchestrator performance?](docs/LOADTEST.md)

## Contributing

We appreciate your interest in contributing to this project! Please refer to the [CONTRIBUTING.md](https://github.com/Azure/GPT-RAG/blob/main/CONTRIBUTING.md) page for detailed guidelines on how to contribute, including information about the Contributor License Agreement (CLA), code of conduct, and the process for submitting pull requests.

Thank you for your support and contributions!

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
