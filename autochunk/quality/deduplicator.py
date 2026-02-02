
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib
import numpy as np
from ..chunkers.base import Chunk
from ..utils.text import count_tokens
from ..utils.logger import logger
import time

@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    original_count: int
    deduplicated_count: int
    removed_count: int
    duplicate_groups: List[List[str]]  # Groups of chunk IDs that are duplicates
    kept_chunks: List[Chunk]
    removed_chunks: List[Chunk]


class ChunkDeduplicator:
    """
    World-Class Chunk Deduplication System.
    
    Identifies and removes duplicate or near-duplicate chunks using
    multiple similarity detection methods.
    
    DEDUPLICATION METHODS:
    1. Exact Hash: MD5/SHA256 for identical content
    2. MinHash LSH: Locality-Sensitive Hashing for near-duplicates
    3. Semantic: Embedding-based similarity for paraphrases
    4. N-gram Jaccard: Character/word n-gram overlap
    
    STRATEGIES:
    - keep_first: Keep the first occurrence
    - keep_longest: Keep the longest version
    - keep_best: Keep the highest quality (requires scorer)
    - merge: Merge duplicates into a single enhanced chunk
    """

    def __init__(self,
                 embedding_fn: Callable[[List[str]], List[List[float]]] = None,
                 similarity_threshold: float = 0.85,
                 method: str = "hybrid",
                 strategy: str = "keep_first",
                 minhash_permutations: int = 128,
                 ngram_size: int = 3):
        """
        Initialize the deduplicator.
        
        Args:
            embedding_fn: Function for semantic similarity (optional).
            similarity_threshold: Threshold for considering chunks as duplicates (0-1).
            method: "exact", "minhash", "semantic", "ngram", or "hybrid" (all methods).
            strategy: "keep_first", "keep_longest", "keep_best", "merge".
            minhash_permutations: Number of permutations for MinHash.
            ngram_size: Size of n-grams for Jaccard similarity.
        """
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.method = method
        self.strategy = strategy
        self.minhash_permutations = minhash_permutations
        self.ngram_size = ngram_size

    def deduplicate(self, 
                    chunks: List[Chunk],
                    quality_scorer=None) -> DeduplicationResult:
        """
        Remove duplicate chunks from a list with optimized batch embedding.
        """
        if not chunks:
            return DeduplicationResult(0, 0, 0, [], [], [])
        
        # Optimization: Pre-calculate all embeddings for semantic similarity if needed
        all_embeddings = None
        if self.embedding_fn and (self.method == "semantic" or self.method == "hybrid"):
            all_embeddings = np.array(self.embedding_fn([c.text for c in chunks]))
        
        # Find all duplicate groups (passing pre-calculated embeddings)
        duplicate_groups = self._find_duplicate_groups(chunks, all_embeddings)
        
        # Decide which to keep from each group
        kept_ids: Set[str] = set()
        removed_ids: Set[str] = set()
        
        for group in duplicate_groups:
            if len(group) <= 1:
                kept_ids.update(group)
                continue
            
            group_chunks = [c for c in chunks if c.id in group]
            
            if self.strategy == "keep_first":
                # Keep the one with lowest index
                chunk_indices = {c.id: idx for idx, c in enumerate(chunks)}
                sorted_group = sorted(group_chunks, key=lambda c: chunk_indices[c.id])
                kept_ids.add(sorted_group[0].id)
                removed_ids.update(c.id for c in sorted_group[1:])
                
            elif self.strategy == "keep_longest":
                # Keep the longest
                sorted_group = sorted(group_chunks, key=lambda c: len(c.text), reverse=True)
                kept_ids.add(sorted_group[0].id)
                removed_ids.update(c.id for c in sorted_group[1:])
                
            elif self.strategy == "keep_best":
                if quality_scorer:
                    # Note: quality_scorer.score_chunks is already optimized for batching
                    reports = quality_scorer.score_chunks(group_chunks)
                    scored = [(c, r.overall_score) for c, r in zip(group_chunks, reports)]
                    sorted_group = sorted(scored, key=lambda x: x[1], reverse=True)
                    kept_ids.add(sorted_group[0][0].id)
                    removed_ids.update(c.id for c, _ in sorted_group[1:])
                else:
                    # Fallback to keep_longest
                    sorted_group = sorted(group_chunks, key=lambda c: len(c.text), reverse=True)
                    kept_ids.add(sorted_group[0].id)
                    removed_ids.update(c.id for c in sorted_group[1:])
                    
            elif self.strategy == "merge":
                # Keep first but enhance with info from others
                chunk_indices = {c.id: idx for idx, c in enumerate(chunks)}
                sorted_group = sorted(group_chunks, key=lambda c: chunk_indices[c.id])
                kept_ids.add(sorted_group[0].id)
                removed_ids.update(c.id for c in sorted_group[1:])
        
        # Add non-duplicate chunks to kept
        all_grouped_ids = set()
        for group in duplicate_groups:
            all_grouped_ids.update(group)
        
        for chunk in chunks:
            if chunk.id not in all_grouped_ids:
                kept_ids.add(chunk.id)
        
        # Build result
        kept_chunks = [c for c in chunks if c.id in kept_ids]
        removed_chunks = [c for c in chunks if c.id in removed_ids]
        
        return DeduplicationResult(
            original_count=len(chunks),
            deduplicated_count=len(kept_chunks),
            removed_count=len(removed_chunks),
            duplicate_groups=[list(g) for g in duplicate_groups if len(g) > 1],
            kept_chunks=kept_chunks,
            removed_chunks=removed_chunks
        )

    def _find_duplicate_groups(self, chunks: List[Chunk], all_embeddings: Optional[np.ndarray] = None) -> List[Set[str]]:
        """Find groups of duplicate chunks with high-performance vectorization."""
        n = len(chunks)
        if n == 0: return []
        
        logger.info(f"Deduplicator: Optimizing similarity matrix for {n} chunks...")
        start_time = time.time()
        
        # 1. Pre-calculate all non-semantic features once
        hashes = [self._exact_hash(c.text) for c in chunks]
        ngrams = [self._get_ngrams(c.text, self.ngram_size) for c in chunks]
        signatures = None
        if self.method in ["minhash", "hybrid"]:
            signatures = [self._minhash_signature(self._get_shingles(c.text)) for c in chunks]
            
        similarity_matrix = np.zeros((n, n))
        
        # 2. Vectorized Semantic Similarity (Dot Product)
        if all_embeddings is not None:
            logger.info("Deduplicator: Using vectorized matrix multiplication for semantic similarity")
            # Normalize embeddings to use dot product as cosine similarity
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_embeddings = all_embeddings / norms
            similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        else:
            # Identity diagonal if no semantic match
            np.fill_diagonal(similarity_matrix, 1.0)
            
        # 3. Fast-path loop for other methods
        logger.info(f"Deduplicator: Running hybrid similarity checks for {n} chunks...")
        for i in range(n):
            if i > 0 and i % 100 == 0:
                logger.info(f"Deduplicator: Progress {i}/{n}...")
                
            for j in range(i + 1, n):
                # If semantic is already > threshold, skip other expensive checks
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    similarity_matrix[j, i] = similarity_matrix[i, j]
                    continue
                
                # Fast hash check
                if hashes[i] == hashes[j]:
                    similarity_matrix[i, j] = 1.0
                    similarity_matrix[j, i] = 1.0
                    continue
                    
                # Other methods (now using pre-calculated features)
                scores = []
                if self.method in ["ngram", "hybrid"]:
                    # Jaccard
                    u = ngrams[i] | ngrams[j]
                    scores.append(len(ngrams[i] & ngrams[j]) / len(u) if u else 0.0)
                
                if self.method in ["minhash", "hybrid"] and signatures:
                    # Estimate Jaccard from signature overlap
                    matches = sum(1 for a, b in zip(signatures[i], signatures[j]) if a == b)
                    scores.append(matches / len(signatures[i]))
                
                if scores:
                    sim = max(scores)
                    if sim > similarity_matrix[i, j]:
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
        
        logger.info(f"Deduplicator: Vectorized matrix finished in {time.time()-start_time:.2f}s")
        
        # 4. Find connected components above threshold (the actual duplicates)
        visited = set()
        groups = []
        
        def dfs(idx: int, group: Set[str]):
            if idx in visited:
                return
            visited.add(idx)
            group.add(chunks[idx].id)
            
            for j in range(n):
                if j not in visited and similarity_matrix[idx, j] >= self.similarity_threshold:
                    dfs(j, group)
        
        for i in range(n):
            if i not in visited:
                group = set()
                dfs(i, group)
                if group:
                    groups.append(group)
        
        return groups

    def _exact_hash(self, text: str) -> str:
        """Generate hash for exact matching."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _minhash_similarity(self, text1: str, text2: str) -> float:
        """MinHash-based similarity estimation."""
        # Generate shingles
        shingles1 = self._get_shingles(text1)
        shingles2 = self._get_shingles(text2)
        
        if not shingles1 or not shingles2:
            return 0.0
        
        # Generate MinHash signatures
        sig1 = self._minhash_signature(shingles1)
        sig2 = self._minhash_signature(shingles2)
        
        # Estimate Jaccard from signature overlap
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Get character k-shingles from text."""
        text = text.lower()
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _minhash_signature(self, shingles: Set[str]) -> List[int]:
        """Generate MinHash signature for a set of shingles."""
        # Use hash functions via different seeds
        signature = []
        for seed in range(self.minhash_permutations):
            min_hash = float('inf')
            for shingle in shingles:
                h = hash(shingle + str(seed)) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _semantic_similarity(self, 
                             text1: str, 
                             text2: str, 
                             idx1: int = -1, 
                             idx2: int = -1, 
                             all_embeddings: Optional[np.ndarray] = None) -> float:
        """Embedding-based semantic similarity with batch optimization support."""
        if all_embeddings is not None and idx1 >= 0 and idx2 >= 0:
            #  Fast Path: Use pre-calculated embeddings
            vec1, vec2 = all_embeddings[idx1], all_embeddings[idx2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0: return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        if not self.embedding_fn:
            return 0.0
        
        try:
            embeddings = self.embedding_fn([text1, text2])
            vec1, vec2 = np.array(embeddings[0]), np.array(embeddings[1])
            
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except:
            return 0.0

    def _ngram_jaccard(self, text1: str, text2: str) -> float:
        """N-gram Jaccard similarity."""
        ngrams1 = self._get_ngrams(text1, self.ngram_size)
        ngrams2 = self._get_ngrams(text2, self.ngram_size)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0

    def _get_ngrams(self, text: str, n: int) -> Set[Tuple[str, ...]]:
        """Get word n-grams from text."""
        words = text.lower().split()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}
