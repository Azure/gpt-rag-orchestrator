#!/bin/bash

# Teste de agentic retrieval com client externo
curl -X POST "http://localhost:9000/orchestrator" \
  -H "Content-Type: application/json" \
  -d '{
    "ask": "What are the copayment amounts for emergency services?",
    "conversation_id": "test-agentic-retrieval-001",
    "type": "nl2sql",
    "user_context": {
      "strategy": "single_agent_rag"
    }
  }' \
  --no-buffer

echo -e "\n\nTeste concluído!"