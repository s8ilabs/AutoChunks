from __future__ import annotations
import os
import requests
from typing import List, Optional
from ..utils.logger import logger
from .base import BaseEncoder

class OpenAIEncoder(BaseEncoder):
    """
    OpenAI Embedding Provider.
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. OpenAI embeddings will fail unless provided.")
        
        self.name = model_name
        # Common dimensions as fallback
        self._dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self._dim = self._dim_map.get(model_name, 1536)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # OpenAI supports batching internally
        payload = {
            "input": texts,
            "model": self.name
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # OpenAI returns data in the same order as input
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI Embedding failed: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self.name
