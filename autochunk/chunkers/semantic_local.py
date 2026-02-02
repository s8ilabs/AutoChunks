
from __future__ import annotations
import numpy as np
from typing import List, Any, Callable
from .base import BaseChunker, Chunk
from ..utils.text import split_sentences, count_tokens

class SemanticLocalChunker(BaseChunker):
    """
     Semantic Chunker using Window-based Gradient Similarity.
    Detects topic shifts by comparing sliding windows of sentence embeddings.
    
    BEST-OF-BREED IMPROVEMENTS:
    1. Windowed-Similarity: Instead of comparing adjacent sentences, it compares the 
       "semantic momentum" of previous and future windows. This suppresses noise from 
       short/outlier sentences.
    2. Dynamic Percentile: Uses a robust percentile-based peak detection for boundaries.
    3. Multi-Factor Safety: Combines semantic drift with strict token caps to ensure LLM 
       context-window compliance.
    """
    name = "semantic_local"

    def chunk(self, 
              doc_id: str, 
              text: str, 
              embedding_fn: Callable[[List[str]], List[List[float]]] = None,
              threshold_percentile: float = 0.9, 
              window_size: int = 3,
              **params) -> List[Chunk]:
        
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(id=f"{doc_id}#sl#0", doc_id=doc_id, text=text, meta={"chunk_index": 0})]

        if embedding_fn is None:
            # Fallback to SentenceAware if no embeddings
            from .sentence_aware import SentenceAwareChunker
            return SentenceAwareChunker().chunk(doc_id, text, **params)

        # 1. Vectorize all sentences -style
        embeddings = np.array(embedding_fn(sentences))
        n = len(embeddings)
        
        # 2. Calculate Windowed Distances (Vectorized Gradient)
        # We use a moving average window to calculate semantic momentum
        def moving_average(a, n=3):
            ret = np.cumsum(a, axis=0)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        # Pre-calculate windowed means (shifted to align with gaps between sentences)
        # For a prompt comparison between windows [i-w+1:i+1] and [i+1:i+w+1]
        
        # We can do this efficiently by calculating all possible window means once
        # and then slicing them.
        
        distances = []
        # Fallback to loop if document is very short, otherwise vectorized logic
        # For simplicity and robustness with edge cases (min/max window), 
        # we'll keep the loop but optimize the inside with pre-calculated norms
        # but in a real-world scenario we'd use np.convolve for a pure vector path.
        
        norms = np.linalg.norm(embeddings, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        norm_embeddings = embeddings / norms[:, np.newaxis]
        
        for i in range(n - 1):
            start_prev = max(0, i - window_size + 1)
            end_prev = i + 1
            start_next = i + 1
            end_next = min(n, i + 1 + window_size)
            
            # Use raw embeddings for mean to preserve magnitude signal if desired,
            # or normalized embeddings for pure cosine.  usually uses raw mean then normalize.
            vec_prev = np.mean(embeddings[start_prev:end_prev], axis=0)
            vec_next = np.mean(embeddings[start_next:end_next], axis=0)
            
            norm_p = np.linalg.norm(vec_prev)
            norm_n = np.linalg.norm(vec_next)
            
            if norm_p < 1e-9 or norm_n < 1e-9:
                distances.append(0.0)
            else:
                sim = np.dot(vec_prev, vec_next) / (norm_p * norm_n)
                distances.append(float(1 - sim))
        
        # 3. Peak Detection
        if not distances:
             return [Chunk(id=f"{doc_id}#sl#0", doc_id=doc_id, text=text, meta={"chunk_index": 0})]
             
        breakpoint_threshold = np.percentile(distances, threshold_percentile * 100)
        
        # 4. Greedy Assembly with Safety Caps
        safety_max_tokens = params.get("base_token_size", 512) * 2 # Usually 2x the target size is a good semantic ceiling
        if "safety_max_tokens" in params:
            safety_max_tokens = params["safety_max_tokens"]

        chunks = []
        curr_buffer = []
        curr_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sent_tokens = count_tokens(sentence)
            
            # Decide to split BEFORE adding the sentence?
            is_semantic_split = False
            if i > 0 and i-1 < len(distances):
                if distances[i-1] >= breakpoint_threshold and distances[i-1] > 0:
                    is_semantic_split = True
            
            # Safety Split: Don't let semantic chunks grow into monsters
            is_safety_split = curr_tokens + sent_tokens > safety_max_tokens
            
            if (is_semantic_split or is_safety_split) and curr_buffer:
                chunks.append(self._make_chunk(doc_id, curr_buffer, len(chunks), "safety" if is_safety_split else "semantic"))
                curr_buffer = []
                curr_tokens = 0
            
            curr_buffer.append(sentence)
            curr_tokens += sent_tokens
            
        if curr_buffer:
            chunks.append(self._make_chunk(doc_id, curr_buffer, len(chunks), "final"))
            
        return chunks

    def _make_chunk(self, doc_id: str, buffer: List[str], index: int, split_reason: str) -> Chunk:
        text = " ".join(buffer).strip()
        return Chunk(
            id=f"{doc_id}#sl#{index}",
            doc_id=doc_id,
            text=text,
            meta={
                "chunk_index": index, 
                "strategy": "semantic_local",
                "split_reason": split_reason,
                "token_count": count_tokens(text)
            }
        )
