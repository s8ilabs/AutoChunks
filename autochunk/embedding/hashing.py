
from __future__ import annotations
from typing import List
import hashlib
import numpy as np

from .base import BaseEncoder

class HashingEmbedding(BaseEncoder):
    """Deterministic, offline-safe feature hashing embedding.
    Not semantically meaningful but good for plumbing tests.
    """
    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "deterministic_hashing"

    def _tok_hash(self, tok: str) -> int:
        return int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        D = self._dim
        for t in texts:
            v = np.zeros(D, dtype=np.float32)
            for tok in t.lower().split():
                h = self._tok_hash(tok)
                idx = h % D
                sign = 1.0 if (h >> 1) & 1 else -1.0
                v[idx] += sign
            # L2 normalize
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v.tolist())
        return vecs
