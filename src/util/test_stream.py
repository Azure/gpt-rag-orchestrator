import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                "http://127.0.0.1:9000/orchestrator",
                json={"ask":"Segun el documento, cual es la diferencia entre model family, model version y model variant?", "conversation_id":None},
                headers={"dapr-api-token": "dev-token"}
            ) as response:
                print(f"Status: {response.status_code}")
                async for chunk in response.aiter_text():
                    print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\nStream failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
