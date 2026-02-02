
from __future__ import annotations
from typing import List, Dict, Any, TYPE_CHECKING, Union
from ..storage.plan import Plan
from ..autochunker import AutoChunker

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode, Document

try:
    from llama_index.core.node_parser import NodeParser, BaseNodeParser
    from llama_index.core.schema import TextNode, BaseNode, Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    class BaseNodeParser: pass
    LLAMA_INDEX_AVAILABLE = False

class AutoChunkLlamaIndexAdapter(BaseNodeParser):
    """
    Official AutoChunks Adapter for LlamaIndex.
    Acts as a native NodeParser for seamless integration into IngestionPipelines.
    """
    def __init__(self, plan: Union[Plan, str]):
        if isinstance(plan, str):
            self.plan = Plan.read(plan)
        else:
            self.plan = plan

        self.chunker = AutoChunker(
            embedding_provider=self.plan.embedding.get("name"),
            embedding_model_or_path=self.plan.embedding.get("model")
        )

    def _parse_nodes(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        """
        Internal implementation for LlamaIndex BaseNodeParser.
        """
        # Convert Nodes to AutoChunks format
        ac_docs = []
        for n in nodes:
            ac_docs.append({
                "id": n.node_id,
                "text": n.get_content(),
                "metadata": n.metadata
            })
        
        # Run the execution pipeline
        gen_name = self.plan.generator_pipeline.get("name")
        params = self.plan.generator_pipeline.get("params", {})
        
        ac_chunks = self.chunker.apply_with_generator(ac_docs, gen_name, params)

        # Convert back to LlamaIndex Nodes
        final_nodes = []
        for ch in ac_chunks:
            node = TextNode(
                text=ch["text"],
                metadata={**ch.get("meta", {}), "autochunk_plan_id": self.plan.id}
            )
            final_nodes.append(node)
            
        return final_nodes

    def get_nodes_from_documents(self, documents: List[Document], **kwargs: Any) -> List[BaseNode]:
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            raise ImportError("Please install llama-index-core: pip install llama-index-core")

        # Convert LlamaIndex docs to AutoChunks format
        ac_docs = []
        for d in documents:
            ac_docs.append({
                "id": d.doc_id,
                "text": d.get_content(),
                "metadata": d.metadata
            })

        # Run the execution pipeline
        gen_name = self.plan.generator_pipeline.get("name")
        params = self.plan.generator_pipeline.get("params", {})
        
        ac_chunks = self.chunker.apply_with_generator(ac_docs, gen_name, params)

        # Convert back to LlamaIndex Nodes
        nodes = []
        for ch in ac_chunks:
            node = TextNode(
                text=ch["text"],
                metadata={**ch.get("meta", {}), "autochunk_plan_id": self.plan.id}
            )
            nodes.append(node)
            
        return nodes
