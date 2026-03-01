import asyncio
import os
import json
import traceback
from dependencies import get_config
from connectors.aifoundry import AIProjectClient
    
async def main():
    try:
        cfg = get_config()
        project_client = AIProjectClient()
        agents_client = project_client.agents_client
        agent_id = cfg.get("AGENT_ID")
        
        print(f"Creating thread for Agent ID: {agent_id}")
        thread = await agents_client.threads.create()
        print(f"Thread ID: {thread.id}")
        
        await agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content="Hello!"
        )
        print("Starting stream...")
        async with await agents_client.runs.stream(
            thread_id=thread.id,
            agent_id=agent_id
        ) as stream:
            async for event_type, event_data, raw in stream:
                print(f"=============================")
                print(f"event_type: {event_type}")
                print(f"type(event_data): {type(event_data)}")
                if event_type == "thread.message.delta":
                    print(f"hasattr(event_data, 'delta'): {hasattr(event_data, 'delta')}")
                    if hasattr(event_data, "delta"):
                        print(f"type(event_data.delta): {type(event_data.delta)}")
                        print(f"hasattr(event_data.delta, 'content'): {hasattr(event_data.delta, 'content')}")
                        if hasattr(event_data.delta, "content") and event_data.delta.content:
                            for block in event_data.delta.content:
                                print(f"block: {block}")
                                if hasattr(block, "text") and block.text and hasattr(block.text, "value"):
                                    print(f"block.text.value: {block.text.value}")
    except Exception as e:
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(main())
