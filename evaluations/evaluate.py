# evaluate.py

import os
from pathlib import Path
from pprint import pprint

import pandas as pd
from fastapi.testclient import TestClient
from azure.identity import (
    ChainedTokenCredential,
    ManagedIdentityCredential,
    AzureCliCredential,
    EnvironmentCredential
)

from azure.ai.projects import AIProjectClient
from azure.ai.evaluation import evaluate, SimilarityEvaluator
from appconfig import AppConfigClient

cfg = AppConfigClient()

# 1) Bring in your FastAPI app
from src.main import app  

# 2) Load Azure App Configuration (label="gpt-rag-orchestrator") into env
cred = ChainedTokenCredential(
    EnvironmentCredential(),
    AzureCliCredential(),
    ManagedIdentityCredential()
)

# 3) Mount TestClient
client = TestClient(app)

# 4) Prepare your AI Project client and SimilarityEvaluator
project = AIProjectClient.from_connection_string(
    conn_str= cfg.get("AI_FOUNDRY_PROJECT_ENDPOINT"),
    credential=cred
)
evaluator_model = {
    "azure_endpoint": cfg.get("AI_FOUNDRY_ACCOUNT_ENDPOINT"),
    "azure_deployment": cfg.get("CHAT_DEPLOYMENT_NAME"),
    "api_version": cfg.get("OPENAI_API_VERSION", "2024-10-21")
}
similarity = SimilarityEvaluator(evaluator_model)

# 5) Wrapper that calls /orchesrator and returns the full text
def evaluate_chat_with_contoso(query: str):
    resp = client.post("/orchestrator", json={"ask": query})
    # TestClient buffers until the stream ends, then .text contains all chunks
    return {"response": resp.text}

# 6) Run evaluate(...) when executed as a script
if __name__ == "__main__":
    # multiprocessing fork workaround
    import multiprocessing, contextlib
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    result = evaluate(
        data="dataset/chat_eval_data.jsonl",
        target=evaluate_chat_with_contoso,
        evaluation_name="evaluate_contoso_similarity",
        evaluators={"similarity": similarity},
        evaluator_config={
            "similarity": {
                "column_mapping": {
                    "query":        "${data.query}",
                    "response":     "${target.response}",
                    "ground_truth":"${data.truth}"
                }
            }
        },        
        azure_ai_project=project.scope,
        output_path="evaluation/evaluation-results.json",
    )

    # show results
    tabular = pd.DataFrame(result["rows"])
    pprint("-----Summarized Metrics-----")
    pprint(result["metrics"])
    pprint("-----Tabular Result-----")
    pprint(tabular)
    pprint(f"View evaluation results in AI Studio: {result['studio_url']}")
