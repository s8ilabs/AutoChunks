
from __future__ import annotations
import numpy as np
from typing import List, Any, Callable, Dict
from .base import BaseChunker, Chunk
from ..utils.text import split_sentences, count_tokens

class HybridSemanticStatChunker(BaseChunker):
    """
     Hybrid Chunker combining Semantic Similarity with Statistical Constraints.
    
    BEST-OF-BREED FEATURES:
    1. Windowed Similarity: Uses window-averaged embeddings for noise suppression.
    2. Percentile-Based Threshold: Adaptive boundary detection like SemanticLocal.
    3. Statistical Forces: Token pressure, sentence length variance, and entropy.
    4. Multi-Factor Scoring: Configurable weights for semantic vs statistical signals.
    """
    name = "hybrid_semantic_stat"

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 window_size: int = 3,
                 threshold_percentile: float = 0.85):
        """
        Initialize the chunker.
        
        Args:
            alpha: Weight for semantic similarity signal (0-1)
            beta: Weight for statistical signal (0-1)
            window_size: Number of sentences for windowed similarity
            threshold_percentile: Percentile for adaptive threshold (0-1)
        """
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile

    def chunk(self, 
              doc_id: str, 
              text: str, 
              embedding_fn: Callable[[List[str]], List[List[float]]] = None,
              alpha: float = None,
              beta: float = None,
              base_token_size: int = 512,
              **params) -> List[Chunk]:
        """
        Split text using hybrid semantic-statistical analysis.
        
        Args:
            doc_id: Document identifier  
            text: Input text
            embedding_fn: Function to generate embeddings for sentences
            alpha: Override semantic weight
            beta: Override statistical weight
            base_token_size: Target chunk size
        
        Returns:
            List of Chunk objects
        """
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(id=f"{doc_id}#hss#0", doc_id=doc_id, text=text, meta={"chunk_index": 0})]

        if embedding_fn is None:
            from .sentence_aware import SentenceAwareChunker
            return SentenceAwareChunker().chunk(doc_id, text, base_token_size=base_token_size)

        # 1. Get embeddings
        embeddings = np.array(embedding_fn(sentences))
        
        # 2. Calculate per-sentence metrics
        sent_lengths = [count_tokens(s) for s in sentences]
        avg_length = np.mean(sent_lengths)
        std_length = np.std(sent_lengths) if len(sent_lengths) > 1 else 1.0
        
        # 3. Calculate windowed semantic distances (Vectorized)
        n = len(embeddings)
        semantic_distances = []
        
        # Pre-calculate norms for efficient similarity calculation
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms < 1e-9] = 1e-9
        
        for i in range(n - 1):
            start_prev = max(0, i - self.window_size + 1)
            end_prev = i + 1
            start_next = i + 1  
            end_next = min(n, i + 1 + self.window_size)
            
            vec_prev = np.mean(embeddings[start_prev:end_prev], axis=0)
            vec_next = np.mean(embeddings[start_next:end_next], axis=0)
            
            norm_p = np.linalg.norm(vec_prev)
            norm_n = np.linalg.norm(vec_next)
            
            if norm_p < 1e-9 or norm_n < 1e-9:
                dist = 0.0
            else:
                sim = np.dot(vec_prev, vec_next) / (norm_p * norm_n)
                dist = float(1 - sim)
            semantic_distances.append(dist)
        
        # 4. Calculate boundary scores (Vectorized Signals)
        semantic_signals = np.array(semantic_distances) if semantic_distances else np.zeros(n-1)
        
        # Vectorize cumulative token pressure
        cumulative_tokens = np.cumsum(sent_lengths)[:-1]
        token_pressures = np.minimum(1.0, (cumulative_tokens / base_token_size) ** 2)
        
        # Vectorize length anomaly signals
        length_zs = np.abs(np.array(sent_lengths[:-1]) - avg_length) / (std_length + 1e-6)
        length_signals = np.minimum(1.0, length_zs / 3)
        
        # Combined statistical signal
        stat_signals = 0.7 * token_pressures + 0.3 * length_signals
        
        # Vectorized combined boundary scores
        combined_scores = (alpha * semantic_signals) + (beta * stat_signals)
        
        # Prepare score_info for the split loop (legacy compatibility with split-logic)
        boundary_scores = []
        for i in range(n - 1):
            boundary_scores.append({
                "position": i,
                "semantic": float(semantic_signals[i]),
                "statistical": float(stat_signals[i]),
                "combined": float(combined_scores[i]),
                "running_tokens": int(cumulative_tokens[i])
            })
        
        # 5. Determine adaptive threshold
        if boundary_scores:
            all_combined = [b["combined"] for b in boundary_scores]
            threshold = np.percentile(all_combined, self.threshold_percentile * 100)
        else:
            threshold = 0.5
        
        # 6. Build chunks using detected boundaries
        chunks = []
        curr_sentences = [sentences[0]]
        curr_tokens = sent_lengths[0]
        
        for i, score_info in enumerate(boundary_scores):
            should_split = False
            split_reason = "none"
            
            # Semantic+Statistical split
            if score_info["combined"] >= threshold:
                should_split = True
                split_reason = "hybrid"
            
            # Safety split (hard token limit)
            next_sent_tokens = sent_lengths[i + 1] if i + 1 < len(sent_lengths) else 0
            if curr_tokens + next_sent_tokens > base_token_size * 1.3:
                should_split = True
                split_reason = "safety"
            
            if should_split and curr_sentences:
                chunk_text = " ".join(curr_sentences)
                chunks.append(Chunk(
                    id=f"{doc_id}#hss#{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    meta={
                        "chunk_index": len(chunks),
                        "strategy": "hybrid_semantic_stat",
                        "split_reason": split_reason,
                        "boundary_score": score_info["combined"],
                        "token_count": count_tokens(chunk_text)
                    }
                ))
                curr_sentences = []
                curr_tokens = 0
            
            # Add next sentence to buffer
            if i + 1 < len(sentences):
                curr_sentences.append(sentences[i + 1])
                curr_tokens += sent_lengths[i + 1]
        
        # Final chunk
        if curr_sentences:
            chunk_text = " ".join(curr_sentences)
            chunks.append(Chunk(
                id=f"{doc_id}#hss#{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                meta={
                    "chunk_index": len(chunks),
                    "strategy": "hybrid_semantic_stat",
                    "split_reason": "final",
                    "token_count": count_tokens(chunk_text)
                }
            ))
        
        return chunks
