APPLICATION_INSIGHTS_CONNECTION_STRING = "APPLICATIONINSIGHTS_CONNECTION_STRING"
APP_NAME = "gpt-rag-orchestrator"# OpenTelemetry service.name uses the Agent Landing Zone prefix (Azure/GPT-RAG#695).
# Audit events keep APP_NAME because their service_name is a versioned contract.
TELEMETRY_SERVICE_NAME = "agentlz.orchestrator"
