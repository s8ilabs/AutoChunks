"""
AutoChunks - World-Class Text Chunking Library

The definitive Swiss Army Knife of text splitting with extreme precision.
"""

from .base import BaseChunker, Chunk

# ============================================================================
# BASIC SPLITTERS
# ============================================================================
from .fixed_length import FixedLengthChunker
from .recursive_character import RecursiveCharacterChunker
from .sentence_aware import SentenceAwareChunker

# ============================================================================
# SEMANTIC SPLITTERS
# ============================================================================
from .semantic_local import SemanticLocalChunker
from .hybrid_semantic_stat import HybridSemanticStatChunker
from .proposition import PropositionChunker
from .agentic import AgenticChunker

# ============================================================================
# STRUCTURE-AWARE SPLITTERS
# ============================================================================
from .layout_aware import LayoutAwareChunker
from .parent_child import ParentChildChunker
from .contextual_retrieval import ContextualRetrievalChunker
from .html_section import HTMLSectionChunker

# ============================================================================
# CODE SPLITTERS
# ============================================================================
from .python_ast import PythonASTChunker

# ============================================================================
# CHUNKER REGISTRY
# ============================================================================
CHUNKER_REGISTRY = {
    # Basic
    'fixed_length': FixedLengthChunker,
    'recursive_character': RecursiveCharacterChunker,
    'sentence_aware': SentenceAwareChunker,
    
    # Semantic
    'semantic_local': SemanticLocalChunker,
    'hybrid_semantic_stat': HybridSemanticStatChunker,
    'proposition': PropositionChunker,
    'agentic': AgenticChunker,
    
    # Structure-Aware
    'layout_aware': LayoutAwareChunker,
    'parent_child': ParentChildChunker,
    'contextual_retrieval': ContextualRetrievalChunker,
    'html_section': HTMLSectionChunker,
    
    # Code
    'python_ast': PythonASTChunker,
}

def get_chunker(name: str) -> BaseChunker:
    """Get a chunker instance by name."""
    if name not in CHUNKER_REGISTRY:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(CHUNKER_REGISTRY.keys())}")
    return CHUNKER_REGISTRY[name]()

def list_chunkers() -> list:
    """List all available chunker names."""
    return list(CHUNKER_REGISTRY.keys())

__all__ = [
    # Base
    'BaseChunker',
    'Chunk',
    
    # Basic
    'FixedLengthChunker',
    'RecursiveCharacterChunker',
    'SentenceAwareChunker',
    
    # Semantic
    'SemanticLocalChunker',
    'HybridSemanticStatChunker',
    'PropositionChunker',
    'AgenticChunker',
    
    # Structure-Aware
    'LayoutAwareChunker',
    'ParentChildChunker',
    'ContextualRetrievalChunker',
    
    # Code
    'PythonASTChunker',
    
    # Utilities
    'CHUNKER_REGISTRY',
    'get_chunker',
    'list_chunkers',
]
