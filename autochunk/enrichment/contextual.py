
from __future__ import annotations
from typing import List, Dict, Any, Callable
from ..utils.logger import logger

class ContextualEnricher:
    """
    Implements Anthropic's 'Contextual Retrieval' concept.
    Prepends a short summary of the parent document to each chunk.
    """
    def __init__(self, summarizer_fn: Callable[[str], str] = None):
        self.summarizer = summarizer_fn

    def enrich_batch(self, chunks: List[Dict[str, Any]], doc_text: str) -> List[Dict[str, Any]]:
        if not self.summarizer:
            logger.warning("No summarizer provided for ContextualEnricher. Skipping.")
            return chunks
        
        try:
            summary = self.summarizer(doc_text)
            for chunk in chunks:
                # Prepend the summary as context
                original_text = chunk["text"]
                chunk["text"] = f"[Document Summary: {summary}]\n\n{original_text}"
                chunk["meta"]["contextual_summary"] = summary
        except Exception as e:
            logger.error(f"Error during contextual enrichment: {e}")
            
        return chunks
