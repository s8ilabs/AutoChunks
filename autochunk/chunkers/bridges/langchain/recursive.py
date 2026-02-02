
from __future__ import annotations
from typing import List
from ...base import BaseChunker, Chunk

try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainRecursiveBridge(BaseChunker):
    name = "langchain_recursive"
    
    def chunk(self, doc_id: str, text: str, base_token_size: int = 512, overlap: int = 64, **params) -> List[Chunk]:
        if not LANGCHAIN_AVAILABLE:
            return []
            
        from ....utils.text import count_tokens
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=base_token_size,
            chunk_overlap=overlap,
            length_function=count_tokens
        )
        
        docs = splitter.split_text(text)
        return [
            Chunk(
                id=f"{doc_id}#lc_rc#{i}",
                doc_id=doc_id,
                text=t,
                meta={"chunk_index": i, "framework": "langchain", "strategy": "recursive"}
            ) for i, t in enumerate(docs)
        ]
