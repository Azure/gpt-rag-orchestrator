# gpt on your data orchestrator

Part of [gpt-rag](https://github.com/Azure/gpt-rag)

## Deploy (quickstart)

Here are the steps to configure cognitive search and deploy ingestion code using the terminal.

**First check your environment meets the requirements**

- You need **[AZ CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)** to log and run Azure commands in the command line.
- You need **Python 3.10** to run the setup script. [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) helps you creating and managing your python environments. 
- **[Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=windows%2Cisolated-process%2Cnode-v4%2Cpython-v2%2Chttp-trigger%2Ccontainer-apps&pivots=programming-language-python#install-the-azure-functions-core-tools)** will be needeed to deploy the chunking function.

**1) Login to Azure** 

run ```az login``` to log into azure. Run ```az login -i``` if using a VM with managed identity to run the setup.

**2) Clone the repo** 

If you plan to customize the ingestion logic, create a new repo by clicking on the **Use this template** button on top of this page.

Clone the repostory locally:  ```git clone https://github.com/azure/gpt-rag-orchestrator```

*If you created a new repository please update the repository URL before running the command*

**3) Adjust your prompt** 

In VSCode you can change the prompt text if you want a custom prompt. 
Use the  ```question_answering.prompt``` located in ```orc/prompts/``` folder, just rembember to keep the *Sources: {sources}* text in the bottom.

**4) Deploy function to Azure** 

Enter in the cloned repo folder: 

```cd gpt-rag-orchestrator```

Use Azure Functions Core Tools to deploy the function: 

```func azure functionapp publish FUNCTION_APP_NAME --python```

After finishing the deployment run the following command to confirm the function was deployed:  

```func azure functionapp list-functions FUNCTION_APP_NAME```

*Replace FUNCTION_APP_NAME with your Orchestrator Function App name before running the command*

**5) Deploy locally (optional)**

With Azure Function extension installed you just need to open ```orc/orchestrator.py``` and "Start Debugging" in VSCode. <br>It will start the server in ```http://localhost:7071/api/orc```.

**References**

- Cognitive search:
[Querying a Vector Index](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query), [REST API Reference](https://learn.microsoft.com/en-us/rest/api/searchservice/preview-api/search-documents) and [Querying Samples](https://github.com/Azure/cognitive-search-vector-pr).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
