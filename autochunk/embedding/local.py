from __future__ import annotations
from typing import List, Dict, Any, Optional
from ..utils.logger import logger
from .base import BaseEncoder

class LocalEncoder(BaseEncoder):
    """
    Local Embedding Engine powered by sentence-transformers.
    Automatically handles GPU/CPU selection and MTEB-aligned pooling logic.
    """
    
    def __init__(self, model_name_or_path: str = "BAAI/bge-small-en-v1.5", device: str = None, cache_folder: str = None, trusted_orgs: List[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers torch")
        
        # Device Detection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        from ..utils.logger import logger
        logger.info(f"Using device [{device.upper()}] for embeddings.")
        
        if device == "cpu":
            # Check if we are missing out on CUDA
            try:
                import subprocess
                nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if nvidia_smi.returncode == 0:
                    logger.warning("Tip: NVIDIA GPU detected hardware-wise, but Torch is using CPU. "
                                   "Suggest reinstalling torch with CUDA support: https://pytorch.org/get-started/locally/")
            except Exception:
                pass
        
        # Safety Check: If downloading, ensure it's from a trusted official source
        if not cache_folder and "/" in model_name_or_path:
            org = model_name_or_path.split("/")[0]
            allowed = trusted_orgs or ["ds4sd", "RapidAI", "BAAI", "sentence-transformers"]
            if org not in allowed:
                raise ValueError(f"Security Alert: Attempting to download from untrusted source '{org}'. "
                                 f"Trusted official orgs are: {allowed}")

        self.name = model_name_or_path
        
        # Check if we might be downloading
        import os
        from ..utils.logger import logger
        
        # Determine the effective cache path
        effective_cache = cache_folder or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_id_folder = "models--" + model_name_or_path.replace("/", "--")
        is_cached = os.path.exists(os.path.join(effective_cache, model_id_folder)) or os.path.exists(model_name_or_path)

        if not is_cached:
            logger.info(f"Network Download: Local model '{model_name_or_path}' not found at {effective_cache}. Starting download from Hugging Face (Official)...")
        else:
            logger.info(f"Cache Hit: Using local model at {effective_cache if not os.path.exists(model_name_or_path) else model_name_or_path}")

        # normalize_embeddings=True is standard for cosine similarity retrieval
        self.model = SentenceTransformer(model_name_or_path, device=device, cache_folder=cache_folder)
        self._dim = self.model.get_sentence_embedding_dimension()
        
        # Log the detected capability
        limit = self.max_seq_length
        logger.info(f"LocalEncoder: Loaded '{model_name_or_path}' with max sequence length: {limit} tokens (truncation @ ~{int(limit * 4 * 0.95)} chars)")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Use internal LRU cache to avoid re-embedding identical text segments
        # across multiple candidate variations (common during hyperparameter sweeps)
        if not hasattr(self, "_cache"):
            self._cache = {}
        
        results = [None] * len(texts)
        to_embed_indices = []
        to_embed_texts = []
        hits = 0
        
        # Determine safe truncation limit
        safe_limit = int(self.max_seq_length * 4 * 0.95)
        
        for i, t in enumerate(texts):
            # Check cache first (using full text as key to avoid collisions)
            if t in self._cache:
                results[i] = self._cache[t]
                hits += 1
            else:
                to_embed_indices.append(i)
                # Truncate strictly for embedding model safety
                # We store original text in cache key for recall, but embed the truncated version? 
                # No, best to cache the truncated text result.
                
                # Actually, if we truncate here, we should be careful.
                # But to save the crash, we must truncate.
                safe_t = t[:safe_limit]
                if len(t) > safe_limit:
                     # Log only once to avoid spam
                     if not hasattr(self, "_logged_truncation"):
                         logger.warning(f"LocalEncoder: Auto-truncating input > {safe_limit} chars for model safety.")
                         self._logged_truncation = True
                
                to_embed_texts.append(safe_t)
        
        if hits > 0:
            logger.info(f"LocalEncoder: Cache Hits={hits}/{len(texts)}")
        
        if to_embed_texts:
            import numpy as np
            # Efficient batching is handled internally by sentence-transformers
            embeddings = self.model.encode(to_embed_texts, show_progress_bar=False)
            
            # Cross-version compatibility: handle both numpy arrays and lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
                
            for relative_idx, emb in enumerate(embeddings):
                # Map back to original global index
                original_idx = to_embed_indices[relative_idx]
                
                # Get ORIGINAL text for cache key (so future calls hit cache)
                original_text = texts[original_idx]
                
                self._cache[original_text] = emb
                results[original_idx] = emb
                
            # Optional: Simple cache eviction if it gets too large (> 10k entries)
            if len(self._cache) > 10000:
                # Clear half the cache if it overflows
                keys = list(self._cache.keys())
                for k in keys[:5000]:
                    del self._cache[k]
                    
        return results

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def max_seq_length(self) -> int:
        """Returns the maximum token length the model can handle."""
        if hasattr(self.model, "max_seq_length"):
            return self.model.max_seq_length
        return 512  # Safe default for BERT-like models
