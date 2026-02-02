
from __future__ import annotations
from typing import List, Dict, Any, Optional
from .base import BaseChunker, Chunk
from .recursive_character import RecursiveCharacterChunker
from ..utils.text import count_tokens

class ParentChildChunker(BaseChunker):
    """
     Hierarchical (Small-to-Big) Parent-Child Chunker.
    
    BEST-OF-BREED FEATURES:
    1. N-Level Hierarchy: Configurable depth (Document > Section > Paragraph > Sentence).
    2. Sibling References: Tracks prev_node_id and next_node_id for traversal.
    3. Parent Context: Stores parent text in metadata for rich LLM context.
    4. Child Overlap: Optional overlap between children for context continuity.
    """
    name = "parent_child"

    def __init__(self, 
                 chunk_sizes: List[int] = None,
                 overlap: int = 0,
                 track_siblings: bool = True):
        """
        Initialize the chunker.
        
        Args:
            chunk_sizes: List of sizes for each hierarchy level.
                         Default [2048, 512, 128] creates 3 levels: large -> medium -> small
            overlap: Token overlap between sibling chunks at each level
            track_siblings: If True, add prev/next references to metadata
        """
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.overlap = overlap
        self.track_siblings = track_siblings

    def chunk(self, 
              doc_id: str, 
              text: str,
              parent_size: int = None,
              child_size: int = None,
              overlap: int = None,
              return_all_levels: bool = False,
              **params) -> List[Chunk]:
        """
        Create hierarchical chunks with parent-child relationships.
        
        Args:
            doc_id: Document identifier
            text: Input text
            parent_size: Override first level size (for backward compatibility)
            child_size: Override last level size (for backward compatibility)
            overlap: Override overlap setting
            return_all_levels: If True, return chunks from all levels, not just leaves
        
        Returns:
            List of Chunk objects (leaf nodes by default, or all nodes if return_all_levels=True)
        """
        # Handle legacy 2-level params
        if parent_size and child_size:
            chunk_sizes = [parent_size, child_size]
        else:
            chunk_sizes = self.chunk_sizes
        
        if overlap is None:
            overlap = self.overlap

        base_chunker = RecursiveCharacterChunker()
        
        all_chunks = []
        
        def _build_hierarchy(input_text: str, 
                            level: int, 
                            parent_info: Dict[str, Any],
                            node_path: str) -> List[Chunk]:
            """
            Recursively build chunk hierarchy.
            
            Args:
                input_text: Text to chunk
                level: Current hierarchy level (0 = root)
                parent_info: Info about parent chunk
                node_path: Path identifier for this node
            
            Returns:
                List of chunks at this level (and below if return_all_levels)
            """
            if level >= len(chunk_sizes):
                return []
            
            current_size = chunk_sizes[level]
            is_leaf = (level == len(chunk_sizes) - 1)
            
            # Create chunks at this level
            level_chunks = base_chunker.chunk(
                doc_id=f"{doc_id}_L{level}",
                text=input_text,
                base_token_size=current_size,
                overlap=overlap
            )
            
            result_chunks = []
            
            for idx, chunk in enumerate(level_chunks):
                chunk_id = f"{node_path}#L{level}#{idx}"
                
                # Build metadata with parent info
                meta = {
                    "chunk_index": idx,
                    "level": level,
                    "is_leaf": is_leaf,
                    "strategy": "parent_child",
                    "token_count": count_tokens(chunk.text)
                }
                
                # Add parent references
                if parent_info:
                    meta["parent_id"] = parent_info.get("id")
                    meta["parent_text"] = parent_info.get("text", "")[:500]  # Truncate for efficiency
                
                # Add sibling references
                if self.track_siblings:
                    if idx > 0:
                        meta["prev_sibling_id"] = f"{node_path}#L{level}#{idx - 1}"
                    if idx < len(level_chunks) - 1:
                        meta["next_sibling_id"] = f"{node_path}#L{level}#{idx + 1}"
                
                node = Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=chunk.text,
                    meta=meta
                )
                
                # Add to results based on return_all_levels setting
                if return_all_levels or is_leaf:
                    result_chunks.append(node)
                
                # Recurse to children if not at leaf level
                if not is_leaf:
                    child_parent_info = {
                        "id": chunk_id,
                        "text": chunk.text
                    }
                    children = _build_hierarchy(
                        chunk.text,
                        level + 1,
                        child_parent_info,
                        chunk_id
                    )
                    
                    # Update parent with child references
                    if children and return_all_levels:
                        node.meta["child_ids"] = [c.id for c in children]
                    
                    result_chunks.extend(children)
            
            return result_chunks
        
        # Build from root
        root_parent_info = {
            "id": doc_id,
            "text": text[:500]  # Document context
        }
        
        all_chunks = _build_hierarchy(text, 0, root_parent_info, doc_id)
        
        # Re-index final chunks sequentially
        for i, chunk in enumerate(all_chunks):
            chunk.meta["global_index"] = i
        
        return all_chunks
