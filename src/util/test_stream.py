import asyncio
import logging
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
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    print(chunk, end="", flush=True)
        except httpx.HTTPError as exc:
            logging.error("Stream request failed (%s)", type(exc).__name__)
            raise SystemExit(1) from None

if __name__ == "__main__":
    asyncio.run(main())
