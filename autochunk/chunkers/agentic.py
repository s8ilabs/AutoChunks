
from __future__ import annotations
from typing import List, Callable, Optional, Any
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens, split_sentences

class AgenticChunker(BaseChunker):
    """
    LLM-Powered Agentic Chunker for Intelligent Boundary Detection.
    
    Uses a language model to decide optimal chunk boundaries based on
    semantic coherence, topic shifts, and content structure.
    
    BEST-OF-BREED FEATURES:
    1. LLM-Decided Boundaries: Model determines where to split based on meaning.
    2. Configurable Prompts: Custom instructions for domain-specific chunking.
    3. Fallback Safety: Reverts to sentence-aware if LLM unavailable.
    4. Batch Processing: Efficient API usage with batched boundary decisions.
    
    Reference: Greg Kamradt's "Agentic Chunker" concept.
    """
    name = "agentic"

    DEFAULT_SYSTEM_PROMPT = """You are a text segmentation expert. Your task is to identify natural boundaries in text where the topic, theme, or focus shifts significantly.

For each boundary you identify, respond with the sentence number (1-indexed) where a NEW section should begin.

Guidelines:
- A new section should start when there's a clear topic shift
- Keep related information together
- Aim for chunks of roughly 3-10 sentences
- Don't split in the middle of a logical argument or explanation
- Consider paragraph breaks as potential (but not mandatory) boundaries"""

    DEFAULT_USER_TEMPLATE = """Analyze the following text and identify where natural section boundaries should occur.

TEXT:
{text}

SENTENCES (numbered):
{numbered_sentences}

Respond with a JSON array of sentence numbers where new sections should BEGIN.
Example: [1, 5, 12, 18] means sections start at sentences 1, 5, 12, and 18.

Only output the JSON array, nothing else."""

    def __init__(self,
                 llm_fn: Callable[[str, str], str] = None,
                 system_prompt: str = None,
                 user_template: str = None,
                 max_sentences_per_call: int = 50):
        """
        Initialize the agentic chunker.
        
        Args:
            llm_fn: Function that takes (system_prompt, user_message) and returns LLM response.
                    If None, uses a mock that falls back to sentence-aware chunking.
            system_prompt: Custom system prompt for the LLM
            user_template: Custom user message template (must include {text} and {numbered_sentences})
            max_sentences_per_call: Max sentences to process in one LLM call
        """
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE
        self.max_sentences_per_call = max_sentences_per_call

    def chunk(self,
              doc_id: str,
              text: str,
              base_token_size: int = 512,
              **params) -> List[Chunk]:
        """
        Use LLM to determine optimal chunk boundaries.
        
        Args:
            doc_id: Document identifier
            text: Input text
            base_token_size: Target chunk size (used as guidance for LLM)
        
        Returns:
            List of Chunk objects
        """
        sentences = split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(id=f"{doc_id}#ag#0", doc_id=doc_id, text=text, 
                         meta={"chunk_index": 0, "strategy": "agentic"})]
        
        # Fallback if no LLM function provided
        if self.llm_fn is None:
            from .sentence_aware import SentenceAwareChunker
            return SentenceAwareChunker().chunk(doc_id, text, base_token_size=base_token_size)
        
        # Get boundary decisions from LLM
        boundaries = self._get_boundaries(sentences)
        
        # Build chunks from boundaries
        chunks = []
        current_start = 0
        
        for boundary in sorted(set(boundaries)):
            if boundary > current_start and boundary <= len(sentences):
                chunk_sentences = sentences[current_start:boundary]
                chunk_text = " ".join(chunk_sentences)
                
                chunks.append(Chunk(
                    id=f"{doc_id}#ag#{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    meta={
                        "chunk_index": len(chunks),
                        "strategy": "agentic",
                        "sentence_range": [current_start, boundary],
                        "token_count": count_tokens(chunk_text)
                    }
                ))
                current_start = boundary
        
        # Add final chunk
        if current_start < len(sentences):
            chunk_sentences = sentences[current_start:]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(Chunk(
                id=f"{doc_id}#ag#{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                meta={
                    "chunk_index": len(chunks),
                    "strategy": "agentic",
                    "sentence_range": [current_start, len(sentences)],
                    "token_count": count_tokens(chunk_text)
                }
            ))
        
        return chunks if chunks else [Chunk(id=f"{doc_id}#ag#0", doc_id=doc_id, text=text,
                                            meta={"chunk_index": 0, "strategy": "agentic"})]

    def _get_boundaries(self, sentences: List[str]) -> List[int]:
        """Get boundary positions from LLM."""
        import json
        
        # Always include position 0 as first boundary
        all_boundaries = [0]
        
        # Process in batches if needed
        for batch_start in range(0, len(sentences), self.max_sentences_per_call):
            batch_end = min(batch_start + self.max_sentences_per_call, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]
            
            # Create numbered list
            numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(batch_sentences)])
            batch_text = " ".join(batch_sentences)
            
            user_message = self.user_template.format(
                text=batch_text,
                numbered_sentences=numbered
            )
            
            try:
                response = self.llm_fn(self.system_prompt, user_message)
                
                # Parse JSON response
                # Handle various response formats
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                
                boundaries = json.loads(response)
                
                # Adjust for batch offset and add to all boundaries
                for b in boundaries:
                    if isinstance(b, int) and b > 0:
                        all_boundaries.append(batch_start + b - 1)  # Convert 1-indexed to 0-indexed
                        
            except Exception as e:
                # On parse failure, use sentence-aware heuristic for this batch
                # Split roughly every 5 sentences
                for i in range(5, len(batch_sentences), 5):
                    all_boundaries.append(batch_start + i)
        
        return sorted(set(all_boundaries))
