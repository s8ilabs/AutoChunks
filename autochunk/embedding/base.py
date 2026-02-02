
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseEncoder(ABC):
    """
     interface for all AutoChunks embedding providers.
    Designed for high-throughput batching and pluggable backends (Local, TEI, OpenAI).
    """
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings into a list of vectors.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the model name/ID.
        """
        pass
