
from __future__ import annotations
from typing import List, Dict, Any, TYPE_CHECKING, Union
from ..storage.plan import Plan
from ..autochunker import AutoChunker, AutoChunkConfig

if TYPE_CHECKING:
    from langchain_core.documents import Document

try:
    from langchain_core.documents import BaseDocumentTransformer, Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    class BaseDocumentTransformer: pass
    LANGCHAIN_AVAILABLE = False

class AutoChunkLangChainAdapter(BaseDocumentTransformer):
    """
    Official AutoChunks Adapter for LangChain.
    Inherits from BaseDocumentTransformer for seamless integration 
    into LangChain Indexing and LCEL pipelines.
    """
    def __init__(self, plan: Union[Plan, str], config: AutoChunkConfig = None):
        if isinstance(plan, str):
            self.plan = Plan.read(plan)
        else:
            self.plan = plan

        # We use a configured AutoChunker to execute the plan
        self.chunker = AutoChunker(
            embedding_provider=self.plan.embedding.get("name"),
            embedding_model_or_path=self.plan.embedding.get("model")
        )

    def transform_documents(self, documents: List[Document], **kwargs: Any) -> List[Document]:
        """
        Apply the optimized AutoChunks plan to a list of LangChain documents.
        This processes ALL documents provided.
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("Please install langchain-core: pip install langchain-core")

        # Convert LangChain docs to AutoChunks format
        ac_docs = []
        for d in documents:
            # We use metadata.get('source', id(d)) as a unique doc_id
            doc_id = str(d.metadata.get("source", id(d)))
            ac_docs.append({
                "id": doc_id,
                "text": d.page_content,
                "metadata": d.metadata
            })

        # Run the execution pipeline
        gen_name = self.plan.generator_pipeline.get("name")
        params = self.plan.generator_pipeline.get("params", {})
        
        ac_chunks = self.chunker.apply_with_generator(ac_docs, gen_name, params)

        # Convert back to LangChain docs
        lc_docs = []
        for ch in ac_chunks:
            # Preserve original metadata and add chunking metadata
            meta = ch.get("meta", {}).copy()
            # If original metadata was passed through, it might be nested or direct
            # For now, we assume simple merger
            lc_docs.append(Document(
                page_content=ch["text"],
                metadata={**meta, "autochunk_plan_id": self.plan.id}
            ))
            
        return lc_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Alias for transform_documents to match TextSplitter interface."""
        return self.transform_documents(documents)

    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.transform_documents(documents)
