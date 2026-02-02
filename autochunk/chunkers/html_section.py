
from __future__ import annotations
from typing import List, Optional, Dict, Any
import re
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

class HTMLSectionChunker(BaseChunker):
    """
    DOM-Aware HTML Chunker for structural web content splitting.
    
    BEST-OF-BREED FEATURES:
    1. DOM-Tree Navigation: Respects HTML hierarchy (doesn't blindly split tags).
    2. Structural Metadata: Tracks DOM path IDs (e.g. body > main > article).
    3. Semantic Grouping: Keeps tables, lists, and definition lists intact.
    4. Header Hierarchy: Uses H1-H6 as natural boundaries.
    """
    name = "html_section"

    # Tags that act as hard section boundaries
    SECTION_TAGS = {'body', 'main', 'section', 'article', 'nav', 'aside', 'footer', 'header'}
    
    # Tags that are logical blocks (like paragraphs)
    BLOCK_TAGS = {'p', 'div', 'blockquote', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'figure', 'li', 'td', 'th'}
    
    # Tags that should be kept atomic if possible
    ATOMIC_TAGS = {'table', 'ul', 'ol', 'dl', 'code', 'pre'}

    def __init__(self, 
                 base_token_size: int = 512, 
                 max_token_size: int = 2048,
                 respect_headers: bool = True):
        self.base_token_size = base_token_size
        self.max_token_size = max_token_size
        self.respect_headers = respect_headers

    def chunk(self, doc_id: str, text: str, **params) -> List[Chunk]:
        """
        Chunk HTML text using DOM analysis.
        
        Args:
            doc_id: Document ID
            text: HTML source string
        """
        if not BS4_AVAILABLE:
            from ..utils.logger import logger
            logger.warning("BeautifulSoup not installed, falling back to RecursiveCharacterChunker")
            from .recursive_character import RecursiveCharacterChunker
            return RecursiveCharacterChunker().chunk(doc_id, text, base_token_size=self.base_token_size)

        # Parse HTML
        soup = BeautifulSoup(text, 'lxml')
        
        # Remove noisy tags
        for t in soup(['script', 'style', 'noscript', 'meta', 'link']):
            t.decompose()

        chunks = []
        current_chunk_text = []
        current_chunk_tokens = 0
        current_meta = {}
        
        # Traverse DOM depth-first
        elements = self._flatten_dom(soup.body if soup.body else soup)
        
        chunk_idx = 0
        
        for elem_text, dom_path, is_header in elements:
            token_count = count_tokens(elem_text)
            
            # If single element is huge, need to split it (fallback)
            if token_count > self.max_token_size:
                # Flush current accumulator first
                if current_chunk_text:
                    self._flush_chunk(doc_id, chunk_idx, current_chunk_text, current_meta, chunks)
                    chunk_idx += 1
                    current_chunk_text = []
                    current_chunk_tokens = 0
                
                # Split the huge element
                from .recursive_character import RecursiveCharacterChunker
                sub_chunks = RecursiveCharacterChunker().chunk(
                    f"{doc_id}_sub", elem_text, base_token_size=self.base_token_size
                )
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        id=f"{doc_id}#html#{chunk_idx}",
                        doc_id=doc_id,
                        text=sc.text,
                        meta={**current_meta, "chunk_index": chunk_idx, "dom_path": dom_path, "subtype": "large_element_split"}
                    ))
                    chunk_idx += 1
                continue
            
            # Check if we should split
            # 1. Header detected (and we have content)
            # 2. Size limit reached
            is_new_section = is_header and self.respect_headers
            is_full = (current_chunk_tokens + token_count) > self.base_token_size
            
            if (is_new_section or is_full) and current_chunk_text:
                self._flush_chunk(doc_id, chunk_idx, current_chunk_text, current_meta, chunks)
                chunk_idx += 1
                current_chunk_text = []
                current_chunk_tokens = 0
                # Use metadata from new starting element (specifically header info)
                current_meta = {"dom_path": dom_path, "is_header": is_header}
            
            if not current_chunk_text:
                current_meta = {"dom_path": dom_path, "is_header": is_header}
            
            current_chunk_text.append(elem_text)
            current_chunk_tokens += token_count
        
        # Final flush
        if current_chunk_text:
            self._flush_chunk(doc_id, chunk_idx, current_chunk_text, current_meta, chunks)
        
        return chunks

    def _flush_chunk(self, doc_id, idx, text_parts, meta, chunks_list):
        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            return
        
        chunks_list.append(Chunk(
            id=f"{doc_id}#html#{idx}",
            doc_id=doc_id,
            text=full_text,
            meta={
                "chunk_index": idx,
                "strategy": "html_section",
                "token_count": count_tokens(full_text),
                **meta
            }
        ))

    def _flatten_dom(self, node) -> List[tuple[str, str, bool]]:
        """
        Flatten DOM into text blocks with metadata.
        Returns list of (text, dom_path, is_header).
        """
        results = []
        
        # Helper to get path
        def get_path(tag):
            path = []
            p = tag.parent
            while p and p.name != '[document]':
                path.insert(0, p.name)
                p = p.parent
            path.append(tag.name)
            return " > ".join(path)

        # Iterate only over block elements or meaningful leaf nodes
        # This is a simplification: complex iteration logic needed for perfect reconstruction
        # We'll walk and extract text from "safe" blocks
        
        for element in node.descendants:
            if isinstance(element, Tag):
                if element.name in self.BLOCK_TAGS or element.name in self.ATOMIC_TAGS:
                    # Check if this element contains OTHER block tags. If so, don't extract yet (wait for children).
                    # Exception: ATOMIC_TAGS (tables, etc.) - extract WHOLE text
                    has_block_children = any(child.name in self.BLOCK_TAGS for child in element.find_all(recursive=False))
                    
                    if element.name in self.ATOMIC_TAGS or not has_block_children:
                        # Extract text
                        text = element.get_text(separator=" ", strip=True)
                        if text:
                            is_header = element.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
                            path = get_path(element)
                            results.append((text, path, is_header))
                            
                        # If atomic, skip processing detailed descendants to avoid dups
                        # (BeautifulSoup yields same nodes if we don't handle this carefully)
                        # Actually node.descendants is a flat generator. 
                        # We need to manually control recursion to avoid duplication.
                        # Since we can't easily skip in .descendants loop, we just rely on the fact 
                        # that we only grab "leaf-like" blocks. 
                        
        # Better approach: recursive generator
        return self._recursive_extract(node)

    def _recursive_extract(self, node, path="") -> List[tuple[str, str, bool]]:
        results = []
        
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                return [(text, path, False)]
            return []
            
        if isinstance(node, Tag):
            new_path = f"{path} > {node.name}" if path else node.name
            is_header = node.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
            
            # Atomic tags: return all text as one block
            if node.name in self.ATOMIC_TAGS:
                text = node.get_text(separator="\n", strip=True)
                if text:
                    return [(text, new_path, is_header)]
                return []
            
            # Block tags: process content
            if node.name in self.BLOCK_TAGS:
                # Process children
                block_content = []
                for child in node.children:
                    block_content.extend(self._recursive_extract(child, new_path))
                
                # If we gathered content, maybe we should join it if it's small?
                # For now, just return specific detailed blocks
                return block_content
                
            # Inline tags (span, b, etc.): just continue
            for child in node.children:
                results.extend(self._recursive_extract(child, path)) # passing parent path for inline
                
        return results
