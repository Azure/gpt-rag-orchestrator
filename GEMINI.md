# GPT-RAG Orchestrator

## Project Overview
This project is the **Orchestrator** component of the **Enterprise RAG (GPT-RAG)** Solution Accelerator. It is a Python-based Azure Functions application that manages the retrieval-augmented generation (RAG) workflow, including conversation management, document processing, and report generation.

**Key Technologies:**
- **Framework:** Azure Functions (Python v2 model), Azure Durable Functions
- **LLM Integration:** LangChain, OpenAI
- **Data Stores:** Azure Cosmos DB, Azure AI Search, Azure Blob Storage
- **Utilities:** WeasyPrint (PDF generation), Tavily (Web scraping)

## Architecture

### Entry Point
- **`function_app.py`**: The main entry point defining Azure Function triggers (HTTP, Timer, Queue, Blob).

### Core Components
- **`orc/`**: Contains the core RAG orchestration logic.
    - `ConversationOrchestrator`: Manages the chat flow, context, and LLM interaction.
    - `unified_orchestrator/`: Likely the newer or consolidated orchestration logic.
- **`orchestrators/`**: Durable Functions orchestrators.
    - `main_orchestrator.py`: Main workflow orchestration.
    - `oneshot_orchestrator.py`: Single-turn interaction.
    - `tenant_orchestrator.py`: Tenant-specific operations.
- **`shared/`**: Common utilities and clients.
    - `cosmos_db.py`, `cosmos_client_async.py`: Database interactions.
    - `blob_utils.py`, `blob_client_async.py`: Storage interactions.
    - `prompts.py`: LLM prompts.
- **`report_worker/`**: Background worker for generating reports (Queue triggered).
- **`webscrapping/`**: Modules for scraping web content (Tavily integration).

### Triggers
- **HTTP:** 
    - `/api/orc`: Main chat endpoint (streaming).
    - `/api/start-orch`: Starts a durable orchestration.
    - `/api/conversations`: Export conversation history.
    - `/api/scrape-page`: Scrape a single URL.
    - `/api/multipage-scrape`: Crawl a website.
- **Timer:** `batch_jobs_timer` (Weekly batch processing).
- **Queue:** `report_worker` (Report generation jobs).
- **Blob:** `blob_trigger` (Document indexing, local dev primarily).

## Setup & Development

### Prerequisites
- Python 3.10+
- Azure Functions Core Tools
- Azurite (for local storage emulation)
- GTK3 (required for WeasyPrint on Windows)

### Installation
1.  **Environment Setup:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    - Copy the template:
        ```bash
        cp local.settings.json.template local.settings.json
        ```
    - Fill in `local.settings.json` with your Azure resource credentials (OpenAI, Cosmos DB, Search, etc.).

### Running Locally
1.  Start Azurite (or ensure it's running).
2.  Start the function app:
    ```bash
    func start
    ```
    The endpoint will be available at `http://localhost:7071/api/orc`.

### Testing
- **Unit/Integration Tests:** Located in `tests/`. Run with `pytest`.
- **API Testing:** A Postman collection is available at `tests/gpt-rag-orchestration.postman_collection.json`.

## Code Standards
- **Formatting:** `black` is used for code formatting.
- **Linting:** `ruff` is used for linting.
- **Type Hints:** Python type hints are encouraged.

## Important Notes
- **Web Scraper:** Uses `weasyprint` which requires GTK3 system dependencies.
- **Legacy Mode:** `ENABLE_LEGACY_QUEUE_WORKER` env var controls the legacy report worker queue trigger.
