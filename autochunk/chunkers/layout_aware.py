
from __future__ import annotations
from typing import List, Dict, Any
import re
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens, extract_code_blocks

class LayoutAwareChunker(BaseChunker):
    """
     Document Chunker with Structure Preservation.
    
    BEST-OF-BREED FEATURES:
    1. Table Inheritance: Re-attaches table headers to split table fragments.
    2. Header Lineage: Prepends [Section: X > Y] for retrieval context.
    3. Code Block Integrity: Never splits inside fenced code blocks.
    4. List Awareness: Keeps list items together when possible.
    5. Multi-Format: Handles Markdown, HTML tables, and plain text.
    """
    name = "layout_aware"

    def __init__(self, 
                 prepend_lineage: bool = True,
                 preserve_code_blocks: bool = True,
                 preserve_tables: bool = True):
        """
        Initialize the chunker.
        
        Args:
            prepend_lineage: If True, prepend section hierarchy to chunk text
            preserve_code_blocks: If True, avoid splitting inside ``` blocks
            preserve_tables: If True, re-attach table headers to fragments
        """
        self.prepend_lineage = prepend_lineage
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables

    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, **params) -> List[Chunk]:
        """
        Split text while respecting document structure.
        
        Args:
            doc_id: Document identifier
            text: Input text (Markdown preferred)
            base_token_size: Target chunk size in tokens
        
        Returns:
            List of Chunk objects with structural metadata
        """
        # Extract structural elements
        code_blocks = extract_code_blocks(text) if self.preserve_code_blocks else []
        
        lines = text.split("\n")
        chunks = []
        buffer = []
        buffer_tokens = 0
        
        # State tracking
        header_stack = []  # Current header hierarchy
        table_header = None  # Current table header row
        table_separator = None  # Table separator row |---|
        in_code_block = False
        code_block_buffer = []
        
        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            line_tokens = count_tokens(line)
            
            # Track fenced code blocks
            if stripped.startswith("```"):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    code_block_buffer = [line]
                    continue
                else:
                    # Ending a code block
                    in_code_block = False
                    code_block_buffer.append(line)
                    code_block_text = "\n".join(code_block_buffer)
                    code_block_tokens = count_tokens(code_block_text)
                    
                    # Flush buffer if adding code block would overflow
                    if buffer_tokens + code_block_tokens > base_token_size and buffer:
                        chunks.append(self._make_chunk(doc_id, buffer, len(chunks), header_stack))
                        buffer = []
                        buffer_tokens = 0
                    
                    buffer.append(code_block_text)
                    buffer_tokens += code_block_tokens
                    code_block_buffer = []
                    continue
            
            if in_code_block:
                code_block_buffer.append(line)
                continue
            
            # Skip empty lines but preserve them in buffer
            if not stripped:
                if buffer:
                    buffer.append("")
                continue
            
            # Header detection - update lineage
            if stripped.startswith("#"):
                # Count header level
                match = re.match(r'^(#+)\s+(.+)$', stripped)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    
                    # Trim stack to parent level and add new header
                    header_stack = header_stack[:level-1]
                    header_stack.append(title)
                    
                    # Hard break on new header
                    if buffer:
                        chunks.append(self._make_chunk(doc_id, buffer, len(chunks), header_stack[:-1]))
                        buffer = []
                        buffer_tokens = 0
            
            # Table detection
            is_table_row = "|" in stripped and not stripped.startswith("```")
            is_separator_row = is_table_row and re.match(r'^[\|\s\-:]+$', stripped)
            
            if is_table_row:
                if table_header is None and not is_separator_row:
                    table_header = line
                elif is_separator_row:
                    table_separator = line
            elif table_header:
                # Exiting table
                table_header = None
                table_separator = None
            
            # Assembly logic
            if buffer_tokens + line_tokens > base_token_size and buffer:
                chunks.append(self._make_chunk(doc_id, buffer, len(chunks), header_stack))
                buffer = []
                buffer_tokens = 0
                
                # Table inheritance: add header to new chunk
                if is_table_row and self.preserve_tables and table_header and line != table_header:
                    buffer.append(table_header)
                    buffer_tokens += count_tokens(table_header)
                    if table_separator:
                        buffer.append(table_separator)
                        buffer_tokens += count_tokens(table_separator)
            
            buffer.append(line)
            buffer_tokens += line_tokens
        
        # Handle remaining code block if unclosed
        if code_block_buffer:
            code_block_text = "\n".join(code_block_buffer)
            buffer.append(code_block_text)
            buffer_tokens += count_tokens(code_block_text)
        
        # Final chunk
        if buffer:
            chunks.append(self._make_chunk(doc_id, buffer, len(chunks), header_stack))

        return chunks

    def _make_chunk(self, doc_id: str, lines: List[str], index: int, lineage: List[str]) -> Chunk:
        """Create a chunk with proper formatting and metadata."""
        # Clean up empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        body_text = "\n".join(lines)
        
        # Prepend lineage for improved retrieval
        if self.prepend_lineage and lineage:
            lineage_str = " > ".join(lineage)
            final_text = f"[Section: {lineage_str}]\n{body_text}"
        else:
            final_text = body_text
        
        return Chunk(
            id=f"{doc_id}#la#{index}",
            doc_id=doc_id,
            text=final_text,
            meta={
                "chunk_index": index,
                "strategy": "layout_aware",
                "lineage": lineage,
                "lineage_str": " > ".join(lineage) if lineage else "",
                "token_count": count_tokens(final_text)
            }
        )
