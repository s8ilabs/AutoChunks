
from __future__ import annotations
from typing import List
from ...base import BaseChunker, Chunk

try:
    from langchain_text_splitters import TokenTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainTokenBridge(BaseChunker):
    name = "langchain_token"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        splitter = TokenTextSplitter(
            chunk_size=base_token_size,
            chunk_overlap=overlap
        )
        
        docs = splitter.split_text(text)
        return [
            Chunk(
                id=f"{doc_id}#lc_tk#{i}",
                doc_id=doc_id,
                text=t,
                meta={"chunk_index": i, "framework": "langchain", "strategy": "token"}
            ) for i, t in enumerate(docs)
        ]
