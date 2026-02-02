
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

class InMemoryIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vecs = []
        self._vec_array = None # Cache for numpy representation
        self.meta = []

    def add(self, vectors: List[List[float]], metas: List[Dict[str, Any]]):
        self.vecs.extend(vectors)
        self.meta.extend(metas)
        self._vec_array = None # Invalidate cache on add

    def search(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        if self._vec_array is None:
            if not self.vecs:
                return []
            self._vec_array = np.array(self.vecs, dtype=np.float32)
        
        V = self._vec_array
        q = np.array(query_vec, dtype=np.float32)
        
        # Matrix multiplication for cosine similarity (if normalized)
        # Handle case where q might be a batch or just a single vector
        if q.ndim == 1:
            sims = (V @ q)
        else:
            sims = (V @ q.T).T
            
        top_k = min(top_k, V.shape[0])
        if q.ndim == 1:
            idxs = np.argpartition(-sims, top_k-1)[:top_k]
            ranked = sorted([(int(i), float(sims[i])) for i in idxs], key=lambda x: -x[1])
        else:
            # Batch mode search support
            results = []
            for query_sims in sims:
                idxs = np.argpartition(-query_sims, top_k-1)[:top_k]
                ranked = sorted([(int(i), float(query_sims[i])) for i in idxs], key=lambda x: -x[1])
                results.append(ranked)
            return results
            
        return ranked
