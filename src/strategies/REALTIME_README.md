# Real-Time Voice Strategy

## Overview

The **Real-Time Voice Strategy** enables natural, two-way voice conversations with AI agents using Azure Communication Services (ACS) and Azure OpenAI's real-time API. This strategy integrates speech recognition, speech synthesis, and retrieval-augmented generation to provide seamless voice interactions over telephone calls.

## Architecture

The real-time voice implementation consists of three main components:

### 1. Real-Time Voice Strategy (`realtime_voice_strategy.py`)

The core strategy class that manages the configuration and behavior of voice-enabled AI agents.

**Key Features:**
- Dynamic prompt management through Jinja2 templates
- Configurable session parameters (temperature, max tokens, tools)
- Action-based request handling for session configuration and prompts
- Integration with the GPT-RAG orchestrator backend

### 2. Real-Time Types (`realtime_types.py`)

Defines the data structures for real-time voice interactions:

```python
class RTActionRequest(BaseModel):
    type: str          # Action type (e.g., "get_prompt", "get_session_config")
    payload: Dict      # Action-specific parameters
```

### 3. Orchestrator Integration

The orchestrator handles real-time voice actions through the `realtime_voice_action_handler` method, which:
- Manages conversation persistence in Cosmos DB
- Routes actions to the appropriate strategy handlers
- Maintains conversation state across interactions

## Configuration

### Session Configuration

Default session settings for real-time voice interactions:

```python
{
    "instructions": "<system_prompt>",
    "temperature": 0.7,
    "max_response_output_tokens": 1024,
    "tool_choice": "auto",  # or "none" if no tools available
    "tools": []             # List of available tools
}
```

### Audio Settings

The real-time client configures audio streaming with:
- **Input Format**: PCM16 (16-bit PCM at 24kHz from ACS)
- **Output Format**: PCM16 (16-bit PCM at 24kHz to ACS)
- **Voice**: Configurable (default: "alloy")
- **Turn Detection**: Server-side VAD with configurable thresholds

### Voice Activity Detection (VAD)

```python
{
    "type": "server_vad",
    "threshold": 0.6,              # Voice detection sensitivity
    "prefix_padding_ms": 300,      # Audio before speech starts
    "silence_duration_ms": 500     # Silence duration to end turn
}
```

## Prompts

Prompts are stored as Jinja2 templates in `src/prompts/realtime/`:

### Available Prompts

1. **default.jinja2** - Main sales/customer service agent
   - Handles customer inquiries about products
   - Provides pricing and product information
   - Maintains professional, concise responses for voice

2. **greet.jinja2** - Initial greeting agent
   - Handles initial customer interactions
   - Routes conversations appropriately

### Prompt Best Practices

When creating voice prompts:
- Keep sentences **short and simple** for natural speech
- Avoid complex sentence structures
- Focus on conversational language
- Include specific instructions about information handling
- Specify DO's and DON'Ts clearly

## Action Handlers

### Get Prompt Action

Retrieves a system prompt by name:

```python
action = RTActionRequest(
    type="get_prompt",
    payload={"name": "default"}
)
```

**Response**: Returns the rendered prompt template as a string.

### Get Session Config Action

Retrieves session configuration:

```python
action = RTActionRequest(
    type="get_session_config",
    payload={"name": "default"}
)
```

**Response**: Returns the complete session configuration dictionary.

## Usage

### Integration with Orchestrator

```python
# Initialize the orchestrator with real-time voice strategy
orchestrator = Orchestrator(
    conversation_id=conversation_id,
    agent_type="rt_voice"
)

# Handle real-time actions
action = RTActionRequest(
    type="get_session_config",
    payload={"name": "default"}
)
result = await orchestrator.realtime_voice_action_handler(action)
```

### Frontend Integration

The real-time voice service (in the `src/` directory) handles:
1. **Incoming/Outbound Calls**: ACS call management
2. **WebSocket Streaming**: Audio streaming between ACS and OpenAI
3. **Transcript Management**: Saving conversation transcripts via orchestrator

## Call Flow

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│   Phone     │◄────►│     ACS      │◄────►│  Realtime   │◄────►│   OpenAI     │
│   Caller    │      │  WebSocket   │      │   Client    │      │  Realtime    │
└─────────────┘      └──────────────┘      └─────────────┘      │     API      │
                                                   │              └──────────────┘
                                                   ▼
                                            ┌─────────────┐
                                            │ Orchestrator│
                                            │  Strategy   │
                                            └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │  Cosmos DB  │
                                            │ (Transcripts)│
                                            └─────────────┘
```

### Workflow

1. **Call Initiation**: Phone call connects to ACS
2. **WebSocket Setup**: ACS establishes media streaming WebSocket
3. **OpenAI Connection**: Realtime client connects to Azure OpenAI
4. **Session Configuration**: Fetches prompts and config from orchestrator
5. **Audio Streaming**: 
   - User audio flows: Phone → ACS → Realtime Client → OpenAI
   - AI audio flows: OpenAI → Realtime Client → ACS → Phone
6. **Transcript Capture**: Both user and assistant transcripts saved to Cosmos DB
7. **Call Cleanup**: Resources released when call ends

## Environment Variables

Required environment variables for the real-time voice system:

```bash
# Orchestrator
ORCHESTRATOR_BASE_URL=http://localhost:9000
ORCHESTRATOR_API_KEY=your-api-key

# Azure Communication Services
ACS_CONNECTION_STRING=your-acs-connection-string
ACS_PHONE_NUMBER=+1234567890
ACS_CALLBACK_URL=https://your-domain.com/api/callbacks
ACS_WEBSOCKET_URL=wss://your-domain.com/api/media

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-realtime-preview
```

## Features

### Voice Conversation Management
- ✅ Real-time speech-to-text and text-to-speech
- ✅ Server-side voice activity detection
- ✅ Configurable AI voice selection
- ✅ Audio format conversion (PCM16)

### Conversation Persistence
- ✅ Automatic transcript capture (user + assistant)
- ✅ Conversation history stored in Cosmos DB
- ✅ Session state management

### Dynamic Configuration
- ✅ Template-based prompt system
- ✅ Runtime session configuration updates
- ✅ Tool integration support (extensible)

### Multi-Channel Support
- ✅ Inbound call handling
- ✅ Outbound call initiation
- ✅ WebSocket-based media streaming

## Extending the Strategy

### Adding New Prompts

1. Create a new Jinja2 template in `src/prompts/realtime/`:
```jinja
You are a helpful assistant specialized in [domain].
Keep responses concise and suitable for voice conversation.
```

2. Access the prompt:
```python
action = RTActionRequest(
    type="get_prompt",
    payload={"name": "your-prompt-name"}
)
```

### Adding Custom Actions

1. Add a new action handler to `RealTimeVoiceStrategy`:
```python
async def handle_custom_action(self, **kwargs) -> dict:
    """Handle custom action logic"""
    # Your implementation
    return {"result": "success"}
```

2. Call the action:
```python
action = RTActionRequest(
    type="custom_action",
    payload={"param1": "value1"}
)
```

### Adding Tools

Configure tools in the session configuration:
```python
self.tools = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Search the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]
```

## Troubleshooting

### Common Issues

**No Audio Stream**
- Verify ACS WebSocket URL is correct
- Check that media streaming is enabled in ACS call setup
- Ensure audio format compatibility (PCM16, 24kHz)

**Prompt Not Loading**
- Confirm Jinja2 template exists in `src/prompts/realtime/`
- Check template syntax is valid
- Verify orchestrator is accessible and API key is correct

**Transcripts Not Saving**
- Validate Cosmos DB connection in orchestrator
- Check conversation_id is properly set
- Verify database container configuration

**Call Connection Issues**
- Confirm ACS_CONNECTION_STRING is valid
- Check callback URL is publicly accessible
- Verify phone number format includes country code

## Performance Considerations

- **Audio Latency**: Typically 200-500ms end-to-end
- **VAD Tuning**: Adjust `threshold` and `silence_duration_ms` for responsiveness
- **Token Management**: Monitor `max_response_output_tokens` for cost control
- **Session Cleanup**: Ensure proper resource cleanup to avoid memory leaks

## Security

- API keys stored in environment variables
- ACS connection strings encrypted
- Transcripts stored with conversation-level isolation
- HTTPS/WSS required for all external communications

## Additional Resources

- [Azure Communication Services Documentation](https://learn.microsoft.com/azure/communication-services/)
- [Azure OpenAI Realtime API Documentation](https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-quickstart)
- [GPT-RAG Main Repository](https://github.com/Azure/gpt-rag)

## Contributing

When contributing to the real-time voice strategy:
1. Follow the existing code structure and naming conventions
2. Add appropriate logging for debugging
3. Update prompts with voice-optimized language
4. Test with actual phone calls (inbound and outbound)
5. Document any new environment variables or configuration options

## License

This project follows the licensing terms of the parent GPT-RAG repository.
