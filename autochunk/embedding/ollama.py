from __future__ import annotations
import requests
from typing import List, Optional
from ..utils.logger import logger
from .base import BaseEncoder

class OllamaEncoder(BaseEncoder):
    """
    Ollama Embedding Provider.
    Assumes Ollama is running locally at http://localhost:11434
    """
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.name = model_name
        self.base_url = base_url.rstrip("/")
        self._dim = None # Will be detected on first call if possible

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embed"
        
        embeddings = []
        # Ollama /api/embed takes one prompt or a list
        payload = {
            "model": self.name,
            "input": texts
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Ollama returns "embeddings" which is a list of vectors
            results = data.get("embeddings", [])
            
            if not results:
                # Fallback to older /api/embeddings if /api/embed is not available or empty
                # /api/embeddings is deprecated but sometimes still used
                 logger.warning("Ollama /api/embed returned no results, trying sequential calls (fallback).")
                 results = []
                 for t in texts:
                    res = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.name, "prompt": t}, timeout=30)
                    res.raise_for_status()
                    results.append(res.json()["embedding"])
            
            if results and self._dim is None:
                self._dim = len(results[0])
                
            return results
        except Exception as e:
            logger.error(f"Ollama Embedding failed: {e}")
            raise

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Try to pulse the model to get dimension
            try:
                self.embed_batch(["pulsing"])
            except:
                return 4096 # common fallback for llama
        return self._dim

    @property
    def model_name(self) -> str:
        return self.name
