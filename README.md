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

The **GPT-RAG Orchestrator** service is an agentic orchestration layer built on Azure AI Foundry Agent Service and the Microsoft Agent Framework. It enables agent-based RAG workflows by coordinating multiple specialized agents—each with a defined role—to collaboratively generate accurate, context-aware responses for complex user queries.

### Available Strategies

| Key | Strategy | Description |
|-----|----------|-------------|
| `single_agent_rag` | Single Agent RAG | RAG strategy using Azure AI Foundry Agent Service v2 with dynamic routing and direct LLM bypass. |
| `maf_agent_service` | MAF + Agent Service | Microsoft Agent Framework with Azure AI Foundry Agent Service v2 for server-side threads and tool orchestration, with request-scoped client lifecycle for stable async cleanup. |
| `maf_lite` | MAF Lite | Microsoft Agent Framework with direct Azure OpenAI model access (no Agent Service dependency). |
| `mcp` | MCP | Request-scoped Microsoft Agent Framework strategy supporting legacy SSE and streamable HTTP MCP servers. |
| `nl2sql` | NL2SQL | Natural language to SQL translation strategy for structured data queries. |

### MCP strategy configuration

The `mcp` strategy now runs on Microsoft Agent Framework instead of Semantic
Kernel. Existing SSE deployments remain compatible: keep `AGENT_STRATEGY=mcp`
and the existing MCP configuration keys. The default transport is still `sse`,
and the streamed response contract is unchanged.

Configure these values in Azure App Configuration. Use the
`gpt-rag-orchestrator` label for an orchestrator-specific override, or the
shared `gpt-rag` label when every component should use the same value:

| Key | Default | Purpose |
| --- | --- | --- |
| `MCP_APP_ENDPOINT` | `http://localhost:80` in Azure; local runs use `http://localhost:5000` | Base URL of the trusted MCP server. HTTPS should be used outside local development. |
| `MCP_SERVER_TRANSPORT` | `sse` | MCP transport. Supported values are `sse` and `streamable_http`. |
| `MCP_CLIENT_TIMEOUT` | `600` | Connection and read timeout in seconds. It must be an integer greater than zero. |
| `MCP_APP_APIKEY` | Unset | Optional API key sent to the MCP server as `X-API-KEY`. Store it as a Key Vault reference, not as plain text. |
| `AGENT_ID` | Unset | Optional agent identifier passed to the request-scoped Microsoft Agent Framework agent. |

The orchestrator resolves the transport endpoint from `MCP_APP_ENDPOINT`:

| Transport | Endpoint suffix | Example resolved endpoint |
| --- | --- | --- |
| `sse` | `/sse` | `https://mcp.contoso.com/sse` |
| `streamable_http` | `/mcp` | `https://mcp.contoso.com/mcp` |

You can configure the base URL or include the matching suffix. The orchestrator
adds the suffix once, so both `https://mcp.contoso.com` and
`https://mcp.contoso.com/sse` resolve to the same SSE endpoint. A conflicting
suffix fails during strategy initialization. For example,
`MCP_SERVER_TRANSPORT=streamable_http` with an endpoint ending in `/sse` is
rejected and the error tells the operator to use `/mcp`.

> [!IMPORTANT]
> The orchestrator forwards the caller's `user-context` header and, when
> configured, `X-API-KEY` to this endpoint for every request. Use only a trusted
> MCP server, require HTTPS outside local development, restrict network access,
> and keep `MCP_APP_APIKEY` in Key Vault.

For a controlled rollout, leave existing deployments on `sse`, deploy the new
orchestrator revision, and verify that a representative request discovers and
calls the expected tools. To adopt streamable HTTP, confirm that the server
exposes `/mcp`, then change the transport and endpoint together. Monitor startup
and request logs for invalid transport, conflicting endpoint suffix, timeout,
or connection errors. Because the configuration names and SSE default are
unchanged, you can roll back to the previous orchestrator image without
rewriting the existing SSE configuration.

### NL2SQL datasource security

> [!IMPORTANT]
> When using the `nl2sql` strategy, configure every SQL Server, Azure SQL, or Fabric SQL datasource with a least-privilege read-only principal. Grant only the `SELECT` permissions needed for approved schemas, tables, or views, and do not use admin, owner, contributor, ingestion, or write-capable identities for NL2SQL query execution. The orchestrator validates generated SQL before execution, but database permissions remain the primary security boundary.

### Retrieval backends

The `rag` and `multimodal_rag` strategies use Foundry IQ's Knowledge Base retrieve API. Alongside the existing native `azureBlob` and `searchIndex` (Pattern B) knowledge sources, the orchestrator can optionally query a Microsoft 365 **Work IQ** knowledge source for grounded answers over the caller's Outlook mail, Teams chats, and SharePoint / OneDrive files.

Work IQ is opt-in and off by default:

- Set `WORK_IQ_ENABLED=true` and `WORK_IQ_KNOWLEDGE_SOURCE_NAME=<your Work IQ knowledge source>` to enable it.
- Work IQ requires a per-user on-behalf-of token. When the OBO token is missing the Work IQ source is skipped with a warning; managed-identity fallback is never used for remote knowledge source kinds.
- ACL is enforced natively by Microsoft 365 via the forwarded user token — no `filterAddOn` is emitted for Work IQ.
- Remote kinds can take 40–60 seconds end-to-end; set `FOUNDRY_IQ_MAX_RUNTIME_SECONDS` (default `120`) to control the retrieve runtime ceiling. The value is only emitted when a remote kind is enabled, so Pattern A / Pattern B requests stay byte-identical.

Work IQ is currently a gated preview and requires admin consent plus a Work IQ knowledge source provisioned on the same Azure AI Search service. See the enablement guide in the [Azure/GPT-RAG](https://github.com/Azure/GPT-RAG) repo for the end-to-end setup ([issue #543](https://github.com/Azure/GPT-RAG/issues/543)).

#### Generic MCP Server knowledge sources (Preview)

Foundry IQ can optionally call one or more preprovisioned generic MCP Server
knowledge sources. The feature is disabled by default. When disabled, the
orchestrator keeps the existing minimal-reasoning `intents` request and does not
add MCP runtime, activity, reasoning, or credential headers.

Enable it with these non-secret App Configuration settings:

```dotenv
RETRIEVAL_BACKEND=foundry_iq
FOUNDRY_IQ_MCP_ENABLED=true
FOUNDRY_IQ_MCP_REASONING_EFFORT=low
FOUNDRY_IQ_MCP_TRUSTED_HOSTS=mcp.contoso.com
FOUNDRY_IQ_MCP_LOG_TOOL_ARGUMENTS=false
FOUNDRY_IQ_MCP_SOURCES_JSON=[{"name":"monitor-mcp","description":"Read-only Azure Monitor MCP source.","serverURL":"https://mcp.contoso.com/mcp","failOnError":true,"maxOutputDocuments":5,"tools":[{"name":"query_logs","outputParsing":{"kind":"json","jsonParameters":{"documentsPath":"$.results[*]","includeContext":true}},"inclusionMode":"reranked","maxOutputTokens":2048}],"queryHeaders":[{"name":"Authorization","valueFrom":{"kind":"managedIdentity","scope":"api://monitor-mcp/.default"}}]}]
```

`serverURL`, tools, and output parsing are registration metadata. Retrieve
requests send only each registered knowledge source name. `queryHeaders` is
non-secret runtime credential metadata and is never rendered into the
top-level Search knowledge-source registration. Its values are resolved per
request and forwarded with Azure AI Search's paired
`<knowledge-source>-header-name[N]` and
`<knowledge-source>-header-value[N]` control headers. Supported `valueFrom.kind`
values are:

- `managedIdentity`, with an explicit downstream `scope`
- `obo`, with an explicit downstream `scope` and an authenticated user request
- `keyVaultSecret`, with a Key Vault `secretName`
- `none`, which forwards no credential

Literal header values and secrets are rejected. MCP hosts require an exact
trusted-host match and HTTPS; query strings, localhost, IP literals, userinfo,
fragments, and reserved hosts are rejected. The Search service `Authorization`
header and document-security `x-ms-query-source-authorization` header remain
separate from MCP credentials.

Enabling MCP switches retrieval to one user `messages` entry with low or medium
reasoning, `extractiveData`, activity diagnostics, and the existing bounded
Foundry IQ runtime/document limits. Required-source activity failures and
credential errors fail closed. Optional-source failures can return successful
references as a partial result and are logged without raw results, credentials,
query strings, or tool arguments by default.

This is a preview contract on API `2026-05-01-preview`. MCP-generated tool
arguments are not guaranteed to be semantically safe. Use read-only tools,
least-privilege scopes, bounded time ranges and row counts, and server-side
argument validation and auditing.

Tool arguments are omitted from normal telemetry. When
`FOUNDRY_IQ_MCP_LOG_TOOL_ARGUMENTS=true`, debug telemetry writes only a
bounded, recursively redacted representation. Credential, authorization,
token, cookie, and header values are always redacted, including paired control
header values. Keep this setting disabled in production unless the remaining
argument data is approved for diagnostic logging.

The source JSON schema is:

| Field | Requirement |
| --- | --- |
| `name` | Required, unique registered knowledge source name. |
| `serverURL` | Required HTTPS registration metadata. Its exact host must appear in `FOUNDRY_IQ_MCP_TRUSTED_HOSTS`; it is never sent in retrieve bodies. |
| `tools` | Required non-empty list with unique tool `name` values. Each tool requires `outputParsing`, `inclusionMode`, and `maxOutputTokens`. `outputParsing` is an object: `{ "kind": "auto" \| "none" \| "split" }`, or `{ "kind": "json", "jsonParameters": { "documentsPath": "...", "includeContext": true } }`. `inclusionMode` only accepts `reranked` or `always`; `maxOutputTokens` must be 1–8192. |
| `failOnError` | Optional, defaults to `true`. A selected source failure fails the request. Set `false` only when partial answers without that source are acceptable. |
| `maxOutputDocuments` | Optional per-source limit from 1–50. The top-level `FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS` still limits final grounding documents. |
| `queryHeaders` | Optional ordered runtime-only metadata. Each entry has a safe `name` and non-secret `valueFrom` metadata; it is not rendered into Search registration. The first resolved header uses the unnumbered pair; later headers use deterministic numeric suffixes. |

For `managedIdentity` and `obo`, `valueFrom.scope` is required.
`keyVaultSecret` requires `valueFrom.secretName`; `none` accepts neither field
and sends no header. MCP retrieval also validates
`FOUNDRY_IQ_MAX_RUNTIME_SECONDS` in the 30–600 range. Invalid enabled
configuration stops retrieval instead of skipping a source.

Track cross-repository provisioning and canonical documentation work in
[Azure/gpt-rag#567](https://github.com/Azure/GPT-RAG/issues/567).

## Documentation

For comprehensive information about GPT-RAG, including architecture details, configuration guides, best practices, troubleshooting resources, deployment guidance, customization options, and advanced usage scenarios, please refer to the [official project documentation](https://azure.github.io/GPT-RAG/).

## Dashboard

The orchestrator ships with an optional admin dashboard mounted at `/dashboard`. It exposes three tabs:

- **Overview**: conversation counts for today, the last 7 days, and the last 30 days; a conversations-over-time chart; average user turns per conversation; and the number of active users.
- **Conversations**: a paginated, newest-first list of conversations across all users, with a detail view that renders the full message history.
- **Configuration**: an allowlisted editor for supported orchestrator settings in Azure App Configuration.

The data comes from the existing conversation/history Cosmos DB container used by the orchestrator (`CONVERSATIONS_DATABASE_CONTAINER` in `DATABASE_NAME`). The dashboard is read-only.

**Enabling the dashboard.** It is disabled by default. Set the App Configuration value `ENABLE_DASHBOARD=true` to mount it. When `ENABLE_DASHBOARD=false` (the default), the `/dashboard` HTML page and every `/api/dashboard/*` route are not registered at all.

**Access control.** When authentication is on (`OAUTH_AZURE_AD_TENANT_ID` is configured), the entire `/api/dashboard/*` surface — except the small `/api/dashboard/version` and `/api/dashboard/auth-config` endpoints used by the SPA at bootstrap — requires the caller's bearer token to include the `Admin` app role. The `/dashboard` HTML page itself is served openly so the SPA can load, call `/api/dashboard/auth-config`, and either sign the user in via MSAL (Authorization Code + PKCE) or render an access-denied state on a 403 response. When authentication is off, the dashboard is open like the rest of the app in development.

**Sign-in configuration.** The SPA reads its runtime auth configuration from `GET /api/dashboard/auth-config`, which is derived from these App Configuration keys under the `gpt-rag-orchestrator` label:

| Key | Required | Purpose |
| --- | --- | --- |
| `OAUTH_AZURE_AD_TENANT_ID` | Yes, to enable the gate | Entra tenant id. When unset, the SPA renders without MSAL and `require_admin` is a no-op. |
| `OAUTH_AZURE_AD_CLIENT_ID` | Yes when tenant is set | Application (client) id of the orchestrator API app registration. |
| `OAUTH_AZURE_AD_API_SCOPE` | Optional | Override for the scope the SPA requests. Defaults to `api://<OAUTH_AZURE_AD_CLIENT_ID>/access_as_user`. |

Use a single Microsoft Entra app registration for both the dashboard SPA and the orchestrator API. The same application (client) id is the MSAL `clientId`, the backend token audience, and the `<client-id>` segment in the default scope `api://<client-id>/access_as_user`. Using separate SPA and API app registrations requires a different configuration model.

The App Registration also needs a **Single-page application** redirect URI pointing at `https://<host>/dashboard/` (trailing slash), an exposed `access_as_user` scope, and an `Admin` app role assigned to every user who should see the dashboard. Full step-by-step in the GPT-RAG docs: [Admin Dashboard Sign-in](https://azure.github.io/GPT-RAG/howto_dashboard_signin/).

Define the role under **Microsoft Entra ID > App registrations > <your app> > App roles**. Create exactly one app role with value `Admin` and allowed member type `Users/Groups`. The backend checks the token's `roles` claim for the exact case-sensitive value `Admin`; other role names or casing are rejected. Assign users under **Enterprise applications > <your app> > Users and groups** by selecting the `Admin` role.

Keep the Enterprise Application setting **Assignment required?** set to **No** when validating the app-level 403 path. With **No**, a signed-in user without the `Admin` app role reaches the dashboard API and receives the dashboard's access-denied state. With **Yes**, Entra blocks the sign-in earlier with `AADSTS50105`, which is useful for strict production gating but does not validate the dashboard's own 403 handling.

After adding or removing the `Admin` role assignment, sign out completely and sign in again. Existing access tokens do not gain or lose `roles` claims retroactively.

Setting only `OAUTH_AZURE_AD_TENANT_ID` is rejected as a server misconfiguration. The dashboard never skips audience validation or silently falls back to unauthenticated mode when auth is partially configured.

**Token scope.** The frontend must request an access token with the orchestrator's own API scope (`api://<client_id>/access_as_user` by default), not a Microsoft Graph scope. App roles are issued in the `roles` claim of an access token only when the token is requested for the application that defines those roles, so a Graph-scoped token will not surface the `Admin` role and every dashboard call will return 403.

**Building the dashboard bundle.** Production builds happen automatically as part of the `Dockerfile` (an MCR base image stage using Node.js 20 runs `npm run build` and copies the static assets into `src/static`). For local development you can run the Vite dev server with hot reload:

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/api` to `http://localhost:9000`, which is where the orchestrator listens locally.

## Prerequisites

Before deploying the application, you must provision the infrastructure as described in the [GPT-RAG](https://github.com/azure/gpt-rag) repo. This includes creating all necessary Azure resources required to support the application runtime.

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

### Deploying the app with azd (recommended)

Initialize the template:
```shell
azd init -t azure/gpt-rag-orchestrator 
```
> [!IMPORTANT]
> Use the **same environment name** with `azd init` as in the infrastructure deployment to keep components consistent.

Update env variables then deploy:
```shell
azd env refresh
azd deploy 
```
> [!IMPORTANT]
> Run `azd env refresh` with the **same subscription** and **resource group** used in the infrastructure deployment.

Aqui está uma versão mais clara, direta e consistente da instrução:

### Deploying the app with a shell script

To deploy using a script, first clone the repository, set the App Configuration endpoint, and then run the deployment script.

##### PowerShell (Windows)

```powershell
git clone https://github.com/Azure/gpt-rag-orchestrator.git
$env:APP_CONFIG_ENDPOINT = "https://<your-app-config-name>.azconfig.io"
cd gpt-rag-orchestrator
.\scripts\deploy.ps1
```

## Found an Issue?

Encountered an error or bug? Help us improve the quality of this accelerator by reporting issues or suggesting enhancements on our **[GitHub Issues page](https://github.com/Azure/GPT-RAG/issues)**. Your feedback helps make GPT-RAG better for everyone!

## Previous Releases

> [!NOTE]  
> For earlier versions, use the corresponding release in the GitHub repository (e.g., v1.0.0 for the initial version).

## 🤝 Contributing

We appreciate contributions! See [CONTRIBUTING](https://github.com/Azure/gpt-rag/blob/main/CONTRIBUTING.md) for guidelines on submitting pull requests.

## Trademarks

This project may contain trademarks or logos. Authorized use of Microsoft trademarks or logos must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Modified versions must not imply sponsorship or cause confusion. Third-party trademarks are subject to their own policies.
