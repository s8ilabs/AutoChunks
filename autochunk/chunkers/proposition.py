
from __future__ import annotations
from typing import List, Callable, Optional
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens, split_sentences

class PropositionChunker(BaseChunker):
    """
    Proposition-Based Chunker for Atomic Fact Extraction.
    
    Instead of arbitrary text splits, this chunker extracts atomic propositions
    (self-contained facts) from the text. Each chunk is a single, verifiable statement.
    
    BEST-OF-BREED FEATURES:
    1. Fact-Level Granularity: Each chunk is one atomic fact.
    2. Self-Contained: Every proposition is understandable without context.
    3. Decontextualized: Pronouns and references are resolved.
    4. LLM-Powered: Uses language model for accurate extraction.
    
    Reference: "Dense X Retrieval" paper, Greg Kamradt's proposition chunker.
    """
    name = "proposition"

    DEFAULT_SYSTEM_PROMPT = """You are an expert at extracting atomic propositions from text.

An atomic proposition is:
- A single, self-contained fact
- Expressed in a complete sentence
- Understandable WITHOUT any additional context
- Has all pronouns replaced with their referents
- Contains no dependent references (like "this", "that", "the above")

For example:
Original: "John went to the store. He bought milk there."
Propositions:
1. John went to the store.
2. John bought milk at the store.

Note how "He" became "John" and "there" became "at the store"."""

    DEFAULT_USER_TEMPLATE = """Extract all atomic propositions from the following text. Each proposition should be:
1. A complete, self-contained sentence
2. Understandable without additional context
3. Have all pronouns resolved to their referents

TEXT:
{text}

Output each proposition on a new line, numbered. Only output the propositions, no other text.

1."""

    def __init__(self,
                 llm_fn: Callable[[str, str], str] = None,
                 system_prompt: str = None,
                 user_template: str = None,
                 max_tokens_per_call: int = 2000):
        """
        Initialize the proposition chunker.
        
        Args:
            llm_fn: Function that takes (system_prompt, user_message) and returns LLM response.
            system_prompt: Custom system prompt
            user_template: Custom user message template (must include {text})
            max_tokens_per_call: Max tokens to process in one LLM call
        """
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE
        self.max_tokens_per_call = max_tokens_per_call

    def chunk(self,
              doc_id: str,
              text: str,
              **params) -> List[Chunk]:
        """
        Extract atomic propositions from text.
        
        Args:
            doc_id: Document identifier
            text: Input text
        
        Returns:
            List of Chunk objects, each containing one proposition
        """
        if not text.strip():
            return []
        
        # Fallback if no LLM function
        if self.llm_fn is None:
            # Use sentence splitting as basic fallback
            sentences = split_sentences(text)
            return [
                Chunk(
                    id=f"{doc_id}#prop#{i}",
                    doc_id=doc_id,
                    text=s.strip(),
                    meta={
                        "chunk_index": i,
                        "strategy": "proposition_fallback",
                        "is_atomic": False
                    }
                ) for i, s in enumerate(sentences) if s.strip()
            ]
        
        # Process text in chunks if too long
        propositions = []
        sentences = split_sentences(text)
        
        current_batch = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = count_tokens(sentence)
            
            if current_tokens + sent_tokens > self.max_tokens_per_call and current_batch:
                # Process current batch
                batch_text = " ".join(current_batch)
                batch_props = self._extract_propositions(batch_text)
                propositions.extend(batch_props)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(sentence)
            current_tokens += sent_tokens
        
        # Process final batch
        if current_batch:
            batch_text = " ".join(current_batch)
            batch_props = self._extract_propositions(batch_text)
            propositions.extend(batch_props)
        
        # Create chunk objects
        chunks = []
        for i, prop in enumerate(propositions):
            if prop.strip():
                chunks.append(Chunk(
                    id=f"{doc_id}#prop#{i}",
                    doc_id=doc_id,
                    text=prop.strip(),
                    meta={
                        "chunk_index": i,
                        "strategy": "proposition",
                        "is_atomic": True,
                        "token_count": count_tokens(prop)
                    }
                ))
        
        return chunks

    def _extract_propositions(self, text: str) -> List[str]:
        """Extract propositions using LLM."""
        user_message = self.user_template.format(text=text)
        
        try:
            response = self.llm_fn(self.system_prompt, user_message)
            
            # Parse numbered list
            propositions = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Remove numbering (1. 2. etc or 1) 2) etc)
                import re
                cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
                if cleaned:
                    propositions.append(cleaned)
            
            return propositions
            
        except Exception as e:
            # Fallback to sentence splitting
            return split_sentences(text)
