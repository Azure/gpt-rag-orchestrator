import aiohttp
import logging

from dependencies import get_config

class SearchClient:
    """
    Encapsulates Azure Cognitive Search queries,
    handling endpoint, API version, token acquisition and retries.
    """
    def __init__(self):
        # ==== Load all config parameters in one place ====
        self.cfg = get_config()
        self.endpoint = self.cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
        self.api_version = self.cfg.get("AZURE_SEARCH_API_VERSION", "2024-07-01")
        self.credential = self.cfg.aiocredential
        # ==== End config block ====

        if not self.endpoint:
            raise ValueError("SEARCH_SERVICE_QUERY_ENDPOINT not set in config")

    async def search(self, index_name: str, body: dict) -> dict:
        """
        Executes a search POST against /indexes/{index_name}/docs/search.
        """
        url = (
            f"{self.endpoint}"
            f"/indexes/{index_name}/docs/search"
            f"?api-version={self.api_version}"
        )

        # get bearer token
        try:
            token = (await self.credential.get_token("https://search.azure.com/.default")).token
        except Exception:
            logging.exception("[search] failed to acquire token")
            raise

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logging.error(f"[search] {resp.status} {text}")
                    raise RuntimeError(f"Search failed: {resp.status} {text}")
                return await resp.json()