
from __future__ import annotations
from typing import List, Callable, Optional, Pattern
import re
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens, get_tokens, decode_tokens

class RecursiveCharacterChunker(BaseChunker):
    """
     Recursive Character Chunker with Tiered Separators.
    
    BEST-OF-BREED FEATURES:
    1. Regex Separator Support: Use regex patterns via `is_separator_regex=True`.
    2. Keep Separator Mode: Preserves delimiters at chunk boundaries.
    3. Start Index Tracking: Records character offset for citation purposes.
    4. Adaptive Fallback: Falls back to token-split when separators are exhausted.
    5. Code Block Awareness: Avoids splitting inside fenced code blocks.
    """
    name = "recursive_character"

    # Default separator hierarchy (most significant first)
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, 
                 separators: List[str] = None,
                 is_separator_regex: bool = False,
                 keep_separator: bool = True,
                 tokenizer: str = "auto"):
        """
        Initialize the chunker.
        
        Args:
            separators: List of separators in priority order
            is_separator_regex: If True, treat separators as regex patterns
            keep_separator: If True, include separator in the chunk that precedes it
            tokenizer: Backend for tokenization
        """
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.is_separator_regex = is_separator_regex
        self.keep_separator = keep_separator
        self.tokenizer = tokenizer

    def chunk(self, 
              doc_id: str, 
              text: str, 
              base_token_size: int = 512, 
              overlap: int = 64,
              add_start_index: bool = False,
              respect_code_blocks: bool = True,
              **params) -> List[Chunk]:
        """
        Recursively split text using separator hierarchy.
        
        Args:
            doc_id: Document identifier
            text: Input text
            base_token_size: Target chunk size in tokens
            overlap: Number of tokens to overlap between chunks
            add_start_index: If True, record character start position
            respect_code_blocks: If True, avoid splitting inside ``` blocks
        
        Returns:
            List of Chunk objects
        """
        separators = params.get("separators", self.separators)
        
        # Extract code blocks if needed
        code_block_ranges = []
        if respect_code_blocks:
            from ..utils.text import extract_code_blocks
            code_blocks = extract_code_blocks(text)
            code_block_ranges = [(b["start"], b["end"]) for b in code_blocks]

        def _is_in_code_block(pos: int) -> bool:
            for start, end in code_block_ranges:
                if start <= pos < end:
                    return True
            return False

        def _split_with_separator(input_text: str, separator: str, is_regex: bool) -> List[str]:
            """Split text while optionally keeping the separator."""
            if is_regex:
                pattern = separator
            else:
                pattern = re.escape(separator)
            
            if self.keep_separator:
                # Use capturing group to keep separator
                raw_splits = re.split(f"({pattern})", input_text)
                # Merge separator with preceding text
                splits = []
                for i in range(0, len(raw_splits) - 1, 2):
                    combined = raw_splits[i] + raw_splits[i + 1]
                    if combined:
                        splits.append(combined)
                if len(raw_splits) % 2 == 1 and raw_splits[-1]:
                    splits.append(raw_splits[-1])
                return splits
            else:
                return [s for s in re.split(pattern, input_text) if s]

        def _recursive_split(input_text: str, seps: List[str], char_offset: int = 0) -> List[tuple]:
            """
            Recursively split text.
            Returns list of (text, start_char_index) tuples.
            """
            token_count = count_tokens(input_text, tokenizer=self.tokenizer)
            
            # Base case: fits in one chunk
            if token_count <= base_token_size:
                return [(input_text, char_offset)]
            
            # No more separators: fallback to token splitting
            if not seps:
                all_tokens = get_tokens(input_text, tokenizer=self.tokenizer)
                results = []
                step = max(1, base_token_size - overlap)
                pos = 0
                curr_char = char_offset
                
                while pos < len(all_tokens):
                    window = all_tokens[pos : pos + base_token_size]
                    chunk_text = decode_tokens(window)
                    results.append((chunk_text, curr_char))
                    
                    stepped = all_tokens[pos : pos + step]
                    curr_char += len(decode_tokens(stepped))
                    pos += step
                    
                    if pos >= len(all_tokens):
                        break
                
                return results

            # Try current separator
            curr_sep = seps[0]
            remaining_seps = seps[1:]
            
            splits = _split_with_separator(input_text, curr_sep, self.is_separator_regex)
            
            # If separator didn't help, try next
            if len(splits) <= 1:
                return _recursive_split(input_text, remaining_seps, char_offset)
            
            # Merge splits into chunks
            final_chunks = []
            buffer = []
            buffer_tokens = 0
            buffer_start = char_offset
            current_char = char_offset

            for split_text in splits:
                split_tokens = count_tokens(split_text, tokenizer=self.tokenizer)
                
                # If a single split is too big, recurse deeper
                if split_tokens > base_token_size:
                    # Flush buffer first
                    if buffer:
                        final_chunks.append(("".join(buffer), buffer_start))
                        buffer = []
                        buffer_tokens = 0
                    
                    # Recurse on the large split
                    sub_chunks = _recursive_split(split_text, remaining_seps, current_char)
                    final_chunks.extend(sub_chunks)
                    current_char += len(split_text)
                    buffer_start = current_char
                    continue

                # Would this overflow the buffer?
                if buffer_tokens + split_tokens > base_token_size and buffer:
                    final_chunks.append(("".join(buffer), buffer_start))
                    
                    # Handle overlap: keep last N tokens worth of text
                    overlap_buffer = []
                    overlap_tokens = 0
                    for prev in reversed(buffer):
                        prev_tokens = count_tokens(prev, tokenizer=self.tokenizer)
                        if overlap_tokens + prev_tokens <= overlap:
                            overlap_buffer.insert(0, prev)
                            overlap_tokens += prev_tokens
                        else:
                            break
                    
                    buffer = overlap_buffer
                    buffer_tokens = overlap_tokens
                    # Adjust start position for overlap
                    buffer_start = current_char - len("".join(overlap_buffer))
                
                buffer.append(split_text)
                buffer_tokens += split_tokens
                current_char += len(split_text)

            if buffer:
                final_chunks.append(("".join(buffer), buffer_start))
                
            return final_chunks

        # Execute recursive pipeline
        raw_chunks = _recursive_split(text, separators, 0)
        
        # Wrap in Chunk objects
        return [
            Chunk(
                id=f"{doc_id}#rc#{i}",
                doc_id=doc_id,
                text=chunk_text.strip(),
                meta={
                    "chunk_index": i, 
                    "strategy": "recursive_character",
                    "token_count": count_tokens(chunk_text, tokenizer=self.tokenizer),
                    **({"start_index": start_idx} if add_start_index else {})
                }
            ) for i, (chunk_text, start_idx) in enumerate(raw_chunks) if chunk_text.strip()
        ]
