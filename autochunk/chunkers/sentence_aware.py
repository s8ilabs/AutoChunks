
from __future__ import annotations
from typing import List, Any
from .base import BaseChunker, Chunk
from ..utils.text import split_sentences, count_tokens

class SentenceAwareChunker(BaseChunker):
    """
     Sentence-Aware Chunker with Look-back Overlap.
    Groups sentences while respecting token limits and providing context continuity.
    
    BEST-OF-BREED IMPROVEMENTS:
    1. Sentence Overlap: Repeating the last N sentences for transition context.
    2. Intelligent Oversize Handling: Uses recursive splitting for "monster sentences" 
       instead of crude fixed-length fallback.
    3. NLTK Integration: Leverages the updated  sentence splitter.
    """
    name = "sentence_aware"

    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        sentences = split_sentences(text)
        chunks = []
        current_buffer = []
        current_tokens = 0
        idx = 0
        
        # We define overlap in sentences if possible, or tokens
        # Standard  is to use token-based sentence overlap
        
        for s in sentences:
            sent_tokens = count_tokens(s)
            
            # 1. Handle "Monster Sentences" (Single sentence > limit)
            if sent_tokens > base_token_size:
                # If we have stuff in buffer, flush first
                if current_buffer:
                    chunks.append(self._make_chunk(doc_id, current_buffer, idx))
                    idx += 1
                    # Prepare overlap from the end of the buffer
                    current_buffer, current_tokens = self._get_overlap(current_buffer, overlap)
                
                # Use recursive logic for this giant sentence
                from .recursive_character import RecursiveCharacterChunker
                sub_chunker = RecursiveCharacterChunker()
                sub_chunks = sub_chunker.chunk(f"{doc_id}_monster", s, base_token_size=base_token_size, overlap=overlap)
                
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        id=f"{doc_id}#sa#{idx}",
                        doc_id=doc_id,
                        text=sc.text,
                        meta={"chunk_index": idx, "strategy": "sentence_aware_recursive"}
                    ))
                    idx += 1
                
                # Reset buffer after a monster sentence break
                current_buffer = []
                current_tokens = 0
                continue

            # 2. Regular Accumulation
            if current_tokens + sent_tokens > base_token_size:
                # Flush
                chunks.append(self._make_chunk(doc_id, current_buffer, idx))
                idx += 1
                
                # Context Overlap (Sentence-Level)
                current_buffer, current_tokens = self._get_overlap(current_buffer, overlap)

            current_buffer.append(s)
            current_tokens += sent_tokens
            
        if current_buffer:
            chunks.append(self._make_chunk(doc_id, current_buffer, idx))
            
        return chunks

    def _get_overlap(self, buffer: List[str], overlap_limit: int) -> tuple[List[str], int]:
        """Calculates the suffix of sentences to carry over for overlap."""
        overlap_buffer = []
        overlap_tokens = 0
        for s in reversed(buffer):
            t = count_tokens(s)
            if overlap_tokens + t <= overlap_limit:
                overlap_buffer.insert(0, s)
                overlap_tokens += t
            else:
                break
        return overlap_buffer, overlap_tokens

    def _make_chunk(self, doc_id: str, buffer: List[str], index: int) -> Chunk:
        text = " ".join(buffer).strip()
        return Chunk(
            id=f"{doc_id}#sa#{index}", 
            doc_id=doc_id, 
            text=text, 
            meta={
                "chunk_index": index, 
                "strategy": "sentence_aware",
                "token_count": count_tokens(text)
            }
        )
