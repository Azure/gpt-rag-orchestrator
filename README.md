# gpt on your data orchestrator

Part of [gpt-rag](https://github.com/Azure/gpt-rag)

## Deploy (quickstart)

**Pre-reqs**

- VS Code with Azure Function App Extension 

**1) Deploy to Azure** 

In VSCode with [Azure Function App Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions) go to the *Azure* Window, reveal your Function App **(fnorc0...)** in the resource explorer, right-click it then select *Deploy*.

**2) Important: Set function key as a secret in Key Vault**

**2.1)** Get the function's **default** key in Azure Portal > Function App (fnorc0...) > App keys > Host keys > default.

![alt text](media/getkey.png)

**2.2)** Set the key as a secret with **orchestratorKey** name in the key vault via Azure Portal > Key Vault (kv0m...) > App keys > Secrets > Generate/Import.

![alt text](media/setsecret.png)

*Note: If you do not have authorization to set secrets, add a Set secret permission for your user.*
<br>*You can do that in Secret permissions in the Access policies option of the Key vault service.*

**3) Deploy locally (optional)**

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
