from enum import Enum
import os
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.bedrock import BedrockChatCompletion
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.vertex_ai import VertexAIChatCompletion
from semantic_kernel.connectors.ai.mistral_ai import MistralAIChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.onnx import OnnxGenAIChatCompletion
from azure.identity.aio import ManagedIdentityCredential, AzureCliCredential, ChainedTokenCredential
from azure.keyvault.secrets.aio import SecretClient as AsyncSecretClient


APIM_ENABLED=os.environ.get("APIM_ENABLED", "False")
APIM_ENABLED = True if APIM_ENABLED.lower() == "true" else False

class ModelProvider(Enum):
    AZURE = "aoai"
    AZURE_INFERENCE = "azure_inference"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AMAZON = "amazon"
    GOOGLE = "google"
    VERTEX = "vertex"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    ONNX = "onnx"

async def create_service(model_provider_str, apim_key=None, credential=None,config=None):
    model_provider=ModelProvider(model_provider_str)
    switch = {
        ModelProvider.AZURE: lambda: create_azure_client(apim_key, credential,config,"aoai"),
        ModelProvider.AZURE_INFERENCE: lambda: create_azure_inference_client("azure_inference"),
        ModelProvider.OPENAI: lambda: create_openai_client("openai"),
        ModelProvider.ANTHROPIC: lambda: create_anthropic_client("anthropic"),
        ModelProvider.AMAZON: lambda: create_amazon_client("amazon"),
        ModelProvider.GOOGLE: lambda: create_google_client("google"),
        ModelProvider.VERTEX: lambda: create_vertex_client("vertex"),
        ModelProvider.MISTRAL: lambda: create_mistral_client("mistral"),
        ModelProvider.OLLAMA: lambda: create_ollama_client("ollama"),
        ModelProvider.ONNX: lambda: create_onnx_client("onnx"),
    }
    
    if model_provider not in switch:
        raise ValueError(f"Unknown model provider: {model_provider}")
    
    return await switch[model_provider]()

async def create_azure_client(apim_key, credential,config,service_id):
    if APIM_ENABLED:
        client = ChatCompletionsClient(
            endpoint=config['model_endpoint'],
            credential=AzureKeyCredential(apim_key),
        )
    else:
        client = ChatCompletionsClient(
            endpoint=config['model_endpoint'],
            credential=credential,
            credential_scopes=["https://cognitiveservices.azure.com/.default"]
        )
    service = AzureAIInferenceChatCompletion(
        service_id=service_id + "_chat_completion",
        ai_model_id=config['deployment'],
        client=client
    )
    return service, client
 
async def create_azure_inference_client(service_id):
    client = DummyAsyncContextManager()
    key=os.getenv("AZURE_AI_INFERENCE_API_KEY")
    if key is None:
        key= await get_secret("azureAIInferenceApiKey")
    model_id=os.getenv("AZURE_AI_INFERENCE_MODEL_ID")
    service = AzureAIInferenceChatCompletion(
        service_id=service_id + "_chat_completion",
        ai_model_id=model_id,
        api_key=key
    )
    return service, client

async def create_openai_client(service_id):
    client = DummyAsyncContextManager()
    key=os.getenv("OPENAI_API_KEY")
    if key is None:
        key=await get_secret("openAIApiKey")
    service = OpenAIChatCompletion(
        api_key=key,
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_anthropic_client(service_id):
    client = DummyAsyncContextManager()
    key=os.getenv("ANTHROPIC_API_KEY")
    if key is None:
        key=await get_secret("anthropicApiKey")
    service = AnthropicChatCompletion(
        api_key=key,
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_amazon_client(service_id):
    client = DummyAsyncContextManager()
    service = BedrockChatCompletion(
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_google_client(service_id):
    client = DummyAsyncContextManager()
    key=os.getenv("GOOGLE_AI_API_KEY")
    if key is None:
        key=await get_secret("googleAiApiKey")
    service = GoogleAIChatCompletion(
        key=key,
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_vertex_client(service_id):
    client = DummyAsyncContextManager()
    service = VertexAIChatCompletion(
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_mistral_client(service_id):
    client = DummyAsyncContextManager()
    key=os.getenv("MISTRAL_API_KEY")
    if key is None:
        key=await get_secret("mistralApiKey")
    service = MistralAIChatCompletion(
        api_key=key,
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_ollama_client(service_id):
    client = DummyAsyncContextManager()
    service = OllamaChatCompletion(
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def create_onnx_client(service_id):
    client = DummyAsyncContextManager()
    service = OnnxGenAIChatCompletion(
        template=os.getenv("ONNX_GEN_AI_CHAT_TEMPLATE"),
        service_id=service_id + "_chat_completion"
    )
    return service, client

async def get_secret(secretName):
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    async with ChainedTokenCredential( ManagedIdentityCredential(), AzureCliCredential()) as credential:
        async with AsyncSecretClient(vault_url=KVUri, credential=credential) as client:
            retrieved_secret = await client.get_secret(secretName)
            value = retrieved_secret.value

    # Consider logging the elapsed_time or including it in the return value if needed
    return value
    




class DummyAsyncContextManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
    
