
from __future__ import annotations
from typing import List, Callable, Optional
from .base import BaseChunker, Chunk

class FixedLengthChunker(BaseChunker):
    """
     Fixed-Length Chunker with Sliding Window (Overlap).
    
    BEST-OF-BREED FEATURES:
    1. Pluggable Length Function: Supports tiktoken, char, word, or custom functions.
    2. Start Index Tracking: Records character offset for citation purposes.
    3. Accurate Token Counting: Uses tiktoken by default for GPT-model accuracy.
    """
    name = "fixed_length"

    def __init__(self, 
                 length_function: Callable[[str], int] = None,
                 tokenizer: str = "auto"):
        """
        Initialize the chunker.
        
        Args:
            length_function: Custom function to measure text length. 
                             If None, uses token counting.
            tokenizer: Backend for tokenization ("auto", "tiktoken", "whitespace", "character")
        """
        self.tokenizer = tokenizer
        self._length_function = length_function

    def _get_length(self, text: str) -> int:
        """Get length using configured method."""
        if self._length_function:
            return self._length_function(text)
        from ..utils.text import count_tokens
        return count_tokens(text, tokenizer=self.tokenizer)

    def chunk(self, 
              doc_id: str, 
              text: str, 
              base_token_size: int = 512, 
              overlap: int = 64,
              add_start_index: bool = False,
              **params) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            doc_id: Document identifier
            text: Input text
            base_token_size: Target chunk size in tokens
            overlap: Number of tokens to overlap between chunks
            add_start_index: If True, record character start position in metadata
        
        Returns:
            List of Chunk objects
        """
        from ..utils.text import get_tokens, decode_tokens, count_tokens
        
        if not text:
            return []
        
        # Get tokens for splitting
        all_tokens = get_tokens(text, tokenizer=self.tokenizer)
        if not all_tokens:
            return []

        chunks = []
        idx = 0
        token_pos = 0
        char_pos = 0  # Track character position for start_index
        
        while token_pos < len(all_tokens):
            # Take a window of tokens
            window_tokens = all_tokens[token_pos : token_pos + base_token_size]
            chunk_text = decode_tokens(window_tokens)
            
            # Build metadata
            meta = {
                "chunk_index": idx, 
                "strategy": "fixed_length",
                "token_count": len(window_tokens)
            }
            
            if add_start_index:
                meta["start_index"] = char_pos
            
            chunks.append(Chunk(
                id=f"{doc_id}#fl#{idx}", 
                doc_id=doc_id, 
                text=chunk_text, 
                meta=meta
            ))
            
            idx += 1
            
            # Step size = window - overlap (move forward)
            step = max(1, base_token_size - overlap)
            
            # Update character position (for start_index tracking)
            stepped_tokens = all_tokens[token_pos : token_pos + step]
            char_pos += len(decode_tokens(stepped_tokens))
            
            token_pos += step
            
            # Boundary safety
            if token_pos >= len(all_tokens):
                break
            
        return chunks
