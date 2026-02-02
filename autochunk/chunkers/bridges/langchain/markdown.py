
from __future__ import annotations
from typing import List
from ...base import BaseChunker, Chunk

try:
    try:
        from langchain_text_splitters import MarkdownTextSplitter
    except ImportError:
        from langchain.text_splitter import MarkdownTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainMarkdownBridge(BaseChunker):
    name = "langchain_markdown"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        from ....utils.text import count_tokens
        splitter = MarkdownTextSplitter(
            chunk_size=base_token_size,
            chunk_overlap=overlap,
            length_function=count_tokens
        )
        
        docs = splitter.split_text(text)
        return [
            Chunk(
                id=f"{doc_id}#lc_md#{i}",
                doc_id=doc_id,
                text=t,
                meta={"chunk_index": i, "framework": "langchain", "strategy": "markdown"}
            ) for i, t in enumerate(docs)
        ]
