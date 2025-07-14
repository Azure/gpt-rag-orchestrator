# evaluations/generate_eval_input.py

import json
import logging
from pathlib import Path
import sys

from fastapi.testclient import TestClient
# Import the FastAPI app; ensure PYTHONPATH includes the 'src' directory
from src.main import app

from azure.search.documents import SearchClient
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential

from appconfig import AppConfigClient

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Uncomment to see DEBUG logs:
    # logging.getLogger().setLevel(logging.DEBUG)

def main():
    setup_logging()
    logger = logging.getLogger("generate_eval_input")

    logger.info("🔧 Loading App Configuration settings")
    cfg = AppConfigClient()
    index_name = cfg.get("SEARCH_RAG_INDEX_NAME")
    search_endpoint = cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
    if not index_name or not search_endpoint:
        logger.error("SEARCH_RAG_INDEX_NAME or SEARCH_SERVICE_QUERY_ENDPOINT not set")
        raise EnvironmentError("SEARCH_RAG_INDEX_NAME and SEARCH_SERVICE_QUERY_ENDPOINT must be set")

    logger.info(f"📚 Connecting to Azure Cognitive Search (index={index_name}, endpoint={search_endpoint})")
    credential = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential
    )

    # Use TestClient on the imported FastAPI app
    client = TestClient(app)

    in_path = Path(__file__).parent.parent / "dataset" / "golden-dataset.jsonl"
    out_path = Path(__file__).parent.parent / "dataset" / "eval-input.jsonl"
    logger.info(f"📄 Reading queries from: {in_path}")
    logger.info(f"📝 Writing eval input to: {out_path}")

    total = sum(1 for _ in open(in_path, encoding="utf-8"))
    logger.info(f"🔢 Total entries to process: {total}")

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            data = json.loads(line)
            query = data["query"]
            truth = data.get("ground-truth") or data.get("truth")  # adapt field name
            logger.info(f"[{idx}/{total}] ▶ Generating response for query: {query!r}")
            # Call the /orchestrator endpoint via TestClient
            resp = client.post("/orchestrator", json={"ask": query, "conversation_id": None})
            response_text = resp.text
            logger.debug(f"[{idx}] Response received (first 100 chars): {response_text[:100]!r}")

            logger.info(f"[{idx}] 🔍 Fetching top-3 documents from Search")
            docs = []
            results = search_client.search(search_text=query, top=3)
            for doc in results:
                content = doc.get("content", "")
                docs.append(content)
            logger.debug(f"[{idx}] Retrieved documents previews: {[d[:50] for d in docs]!r}")

            # Build record without 'item' field
            record = {
                "query": query,
                "truth": truth,
                "response": response_text,
                "context": docs
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"[{idx}/{total}] ✔ Record written")

    logger.info("✅ Eval input generation complete.")

if __name__ == "__main__":
    main()
