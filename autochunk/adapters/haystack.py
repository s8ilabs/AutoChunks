
from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from ..storage.plan import Plan
from ..autochunker import AutoChunker

try:
    from haystack import component, Document
    HAYSTACK_AVAILABLE = True
except ImportError:
    # Robust fallback for environment without Haystack
    def component(cls): return cls
    def output_types(**kwargs):
        def decorator(func): return func
        return decorator
    component.output_types = output_types
    class Document: pass
    HAYSTACK_AVAILABLE = False

@component
class AutoChunkHaystackAdapter:
    """
    Official AutoChunks Adapter for Haystack 2.0.
    Acts as a Pipeline Component for optimized document splitting.
    """
    def __init__(self, plan: Union[Plan, str]):
        if isinstance(plan, str):
            self.plan = Plan.read(plan)
        else:
            self.plan = plan
        
        # Initialize internal engine
        self.chunker = AutoChunker(
            embedding_provider=self.plan.embedding.get("name"),
            embedding_model_or_path=self.plan.embedding.get("model")
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Implementation of the Haystack Component interface.
        """
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Please install haystack-ai: pip install haystack-ai")

        # Convert Haystack docs to AutoChunks format
        ac_docs = []
        for d in documents:
            ac_docs.append({
                "id": str(getattr(d, "id", hash(d.content))),
                "text": d.content,
                "metadata": d.meta
            })

        # Process via  pipeline
        gen_name = self.plan.generator_pipeline.get("name")
        params = self.plan.generator_pipeline.get("params", {})
        ac_chunks = self.chunker.apply_with_generator(ac_docs, gen_name, params)

        # Re-wrap as Haystack Documents
        return {
            "documents": [
                Document(
                    content=ch["text"],
                    meta={**ch.get("meta", {}), "autochunk_plan_id": self.plan.id}
                ) for ch in ac_chunks
            ]
        }
